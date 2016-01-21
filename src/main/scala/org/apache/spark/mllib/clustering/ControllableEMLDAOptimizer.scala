package org.apache.spark.mllib.clustering

import java.util.Random

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, normalize}

import org.apache.spark.graphx.impl.GraphImpl
import org.apache.spark.graphx.{Edge, Graph, _}
import org.apache.spark.mllib.impl.PeriodicGraphCheckpointer
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD

sealed trait ControllableLDAOptimizer {

  /**
   * Initializer for the optimizer. LDA passes the common parameters to the optimizer and
   * the internal structure can be initialized properly.
   */
  private[clustering] def initialize(docs: RDD[(Long, Vector)], lda: LDA): ControllableLDAOptimizer

  private[clustering] def next(): ControllableLDAOptimizer

  private[clustering] def getLDAModel(iterationTimes: Array[Double]): LDAModel
}

  final class ControllableEMLDAOptimizer extends ControllableLDAOptimizer {

  import LDA._

    /**
      * The following fields will only be initialized through the initialize() method
      */
    private[clustering] var graph: Graph[TopicCounts, TokenCount] = null
    private[clustering] var k: Int = 0
    private[clustering] var vocabSize: Int = 0
    private[clustering] var docConcentration: Double = 0
    private[clustering] var topicConcentration: Double = 0
    private[clustering] var checkpointInterval: Int = 10
    private var graphCheckpointer: PeriodicGraphCheckpointer[TopicCounts, TokenCount] = null

    /**
      * Compute bipartite term/doc graph.
      */
    override private[clustering] def initialize(docs: RDD[(Long, Vector)], lda: LDA): ControllableEMLDAOptimizer = {
      // EMLDAOptimizer currently only supports symmetric document-topic priors
      val docConcentration = lda.getDocConcentration

      val topicConcentration = lda.getTopicConcentration
      val k = lda.getK

      // Note: The restriction > 1.0 may be relaxed in the future (allowing sparse solutions),
      // but values in (0,1) are not yet supported.
      require(docConcentration > 1.0 || docConcentration == -1.0, s"LDA docConcentration must be" +
        s" > 1.0 (or -1 for auto) for EM Optimizer, but was set to $docConcentration")
      require(topicConcentration > 1.0 || topicConcentration == -1.0, s"LDA topicConcentration " +
        s"must be > 1.0 (or -1 for auto) for EM Optimizer, but was set to $topicConcentration")

      this.docConcentration = if (docConcentration == -1) (50.0 / k) + 1.0 else docConcentration
      this.topicConcentration = if (topicConcentration == -1) 1.1 else topicConcentration
      val randomSeed = lda.getSeed

      // For each document, create an edge (Document -> Term) for each unique term in the document.
      val edges: RDD[Edge[TokenCount]] = docs.flatMap { case (docID: Long, termCounts: Vector) =>
        // Add edges for terms with non-zero counts.
        termCounts.toBreeze.activeIterator.filter(_._2 != 0.0).map { case (term, cnt) =>
          Edge(docID, term2index(term), cnt)
        }
      }

      // Create vertices.
      // Initially, we use random soft assignments of tokens to topics (random gamma).
      val docTermVertices: RDD[(VertexId, TopicCounts)] = {
        val verticesTMP: RDD[(VertexId, TopicCounts)] =
          edges.mapPartitionsWithIndex { case (partIndex, partEdges) =>
            val random = new Random(partIndex + randomSeed)
            partEdges.flatMap { edge =>
              val gamma = normalize(BDV.fill[Double](k)(random.nextDouble()), 1.0)
              val sum = gamma * edge.attr
              Seq((edge.srcId, sum), (edge.dstId, sum))
            }
          }
        verticesTMP.reduceByKey(_ + _)
      }

      // Partition such that edges are grouped by document
      this.graph = Graph(docTermVertices, edges).partitionBy(PartitionStrategy.EdgePartition1D)
      this.k = k
      this.vocabSize = docs.take(1).head._2.size
      this.checkpointInterval = lda.getCheckpointInterval
      this.graphCheckpointer = new PeriodicGraphCheckpointer[TopicCounts, TokenCount](
        checkpointInterval, graph.vertices.sparkContext)
      this.graphCheckpointer.update(this.graph)
      this.globalTopicTotals = computeGlobalTopicTotals()
      this
    }

    override private[clustering] def next(): ControllableEMLDAOptimizer = {
      require(graph != null, "graph is null, EMLDAOptimizer not initialized.")

      val eta = topicConcentration
      val W = vocabSize
      val alpha = docConcentration

      val N_k = globalTopicTotals
      val sendMsg: EdgeContext[TopicCounts, TokenCount, (Boolean, TopicCounts)] => Unit =
        (edgeContext) => {
          // Compute N_{wj} gamma_{wjk}
          val N_wj = edgeContext.attr
          // E-STEP: Compute gamma_{wjk} (smoothed topic distributions), scaled by token count
          // N_{wj}.
          val scaledTopicDistribution: TopicCounts =
            computePTopic(edgeContext.srcAttr, edgeContext.dstAttr, N_k, W, eta, alpha) *= N_wj
          edgeContext.sendToDst((false, scaledTopicDistribution))
          edgeContext.sendToSrc((false, scaledTopicDistribution))
        }
      // The Boolean is a hack to detect whether we could modify the values in-place.
      // TODO: Add zero/seqOp/combOp option to aggregateMessages. (SPARK-5438)
      val mergeMsg: ((Boolean, TopicCounts), (Boolean, TopicCounts)) => (Boolean, TopicCounts) =
        (m0, m1) => {
          val sum =
            if (m0._1) {
              m0._2 += m1._2
            } else if (m1._1) {
              m1._2 += m0._2
            } else {
              m0._2 + m1._2
            }
          (true, sum)
        }
      // M-STEP: Aggregation computes new N_{kj}, N_{wk} counts.
      val docTopicDistributions: VertexRDD[TopicCounts] =
        graph.aggregateMessages[(Boolean, TopicCounts)](sendMsg, mergeMsg)
          .mapValues(_._2)
      // Update the vertex descriptors with the new counts.
      val newGraph = GraphImpl.fromExistingRDDs(docTopicDistributions, graph.edges)
      graph = newGraph
      graphCheckpointer.update(newGraph)
      globalTopicTotals = computeGlobalTopicTotals()
      this
    }

    /**
      * Aggregate distributions over topics from all term vertices.
      *
      * Note: This executes an action on the graph RDDs.
      */
    private[clustering] var globalTopicTotals: TopicCounts = null

    private def computeGlobalTopicTotals(): TopicCounts = {
      val numTopics = k
      graph.vertices.filter(isTermVertex).values.fold(BDV.zeros[Double](numTopics))(_ += _)
    }

    override private[clustering] def getLDAModel(iterationTimes: Array[Double]): LDAModel = {
      require(graph != null, "graph is null, EMLDAOptimizer not initialized.")
      this.graphCheckpointer.deleteAllCheckpoints()
      // The constructor's default arguments assume gammaShape = 100 to ensure equivalence in
      // LDAModel.toLocal conversion
      new DistributedLDAModel(this.graph, this.globalTopicTotals, this.k, this.vocabSize,
        Vectors.dense(Array.fill(this.k)(this.docConcentration)), this.topicConcentration,
        iterationTimes)
    }
  }

