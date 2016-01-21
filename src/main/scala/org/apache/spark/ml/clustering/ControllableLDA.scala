package org.apache.spark.ml.clustering

import java.util.Random

import breeze.linalg.DenseVector

import org.apache.spark.graphx._
import org.apache.spark.graphx.impl.GraphImpl
import org.apache.spark.ml.tuning.bandit.Controllable
import org.apache.spark.mllib.clustering.{DistributedLDAModel => OldDistributedLDAModel, LocalLDAModel => OldLocalLDAModel, LDA => OldLDA, LDAOptimizer, DistributedLDAModel, LDAModel, LDA}
import org.apache.spark.mllib.impl.PeriodicGraphCheckpointer
import org.apache.spark.mllib.linalg.{Vectors, Vector}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame

class ControllableLDA extends LDA with Controllable {

  def setInitialModel(model: LDAModel): this.type = set(initialModel, Some(model))

  override def fit(dataset: DataFrame): LDAModel = {
    transformSchema(dataset.schema, logging = true)
    val oldLDA = new OldLDA()
      .setK($(k))
      .setDocConcentration(getOldDocConcentration)
      .setTopicConcentration(getOldTopicConcentration)
      .setMaxIterations($(maxIter))
      .setSeed($(seed))
      .setCheckpointInterval($(checkpointInterval))
      .setOptimizer(getOldOptimizer)
    // TODO: persist here, or in old LDA?
    val oldData = LDA.getOldDataset(dataset, $(featuresCol))
    val oldModel = oldLDA.run(oldData)
    val newModel = oldModel match {
      case m: OldLocalLDAModel =>
        new LocalLDAModel(uid, m.vocabSize, m, dataset.sqlContext)
      case m: OldDistributedLDAModel =>
        new DistributedLDAModel(uid, m.vocabSize, m, dataset.sqlContext, None)
    }
    copyValues(newModel).setParent(this)
  }


}
