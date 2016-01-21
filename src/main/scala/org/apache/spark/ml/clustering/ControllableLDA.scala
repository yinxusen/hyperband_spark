package org.apache.spark.ml.clustering

import org.apache.spark.ml.tuning.bandit.Controllable
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.clustering.{DistributedLDAModel => OldDistributedLDAModel, LDA => OldLDA, LocalLDAModel => OldLocalLDAModel}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame

class ControllableLDA extends LDA with Controllable {

  def setInitialModel(model: LDAModel): this.type = set(initialModel, Some(model))

  var mllibLDA: Option[(OldLDA, RDD[(Long, Vector)])] = None

  override def fit(dataset: DataFrame): LDAModel = {
    transformSchema(dataset.schema, logging = true)

    val (oldLDA, oldData) = if (mllibLDA.isDefined) {
      mllibLDA.get
    } else {
      (new OldLDA()
        .setK($(k))
        .setDocConcentration(getOldDocConcentration)
        .setTopicConcentration(getOldTopicConcentration)
        .setMaxIterations($(maxIter))
        .setSeed($(seed))
        .setCheckpointInterval($(checkpointInterval))
        .setOptimizer(getOldOptimizer),
        LDA.getOldDataset(dataset, $(featuresCol)))
    }

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
