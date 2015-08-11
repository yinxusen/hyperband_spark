package org.apache.spark.ml.tuning.bandit

import org.apache.spark.ml.feature.StandardScaler
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.{DataFrame, Row, SQLContext}

/**
 * Created by panda on 8/9/15.
 */
abstract class Dataset(
    val name: String,
    val trainingSet: DataFrame,
    val validationSet: DataFrame,
    val testSet: DataFrame) {
  def baseValErr(): Double
}

class ClassifyDataset(
    override val name: String,
    override val trainingSet: DataFrame,
    override val validationSet: DataFrame,
    override val testSet: DataFrame)
  extends Dataset(name, trainingSet, validationSet, testSet) {

  override def baseValErr(): Double = {
    val ySum = validationSet.select("label").map { case Row(x: Double) => x}.sum()
    val xCnt = validationSet.count()
    0.5 - math.abs(ySum) / (2 * xCnt)
  }
}

object ClassifyDataset {
  def scaleAndPartitionData(
      sqlCtx: SQLContext,
      dataName: String,
      fileName: String,
      normalizeY: Boolean = false): ClassifyDataset = {
    val sc = sqlCtx.sparkContext
    val data = MLUtils.loadLibSVMFile(sc, fileName)
    val (trainingAndValidation, test) = Utils.splitTrainTest(data, 0.1, 0)
    val (training, validation) = Utils.splitTrainTest(trainingAndValidation, 0.2, 0)

    val trainingDF = sqlCtx.createDataFrame(training)
    val validationDF = sqlCtx.createDataFrame(validation)
    val testDF = sqlCtx.createDataFrame(test)

    val standardScaler = new StandardScaler().setWithMean(true).setWithStd(true)
      .setInputCol("features").setOutputCol("scaledFeatures")

    val xScaler = standardScaler.fit(trainingDF)

    val scaledTraining = xScaler.transform(trainingDF)
    val scaledValidation = xScaler.transform(validationDF)
    val scaledTest = xScaler.transform(testDF)

    if (normalizeY) {
      val yScaler = standardScaler.setInputCol("label").setOutputCol("scaledLabel")
        .fit(scaledTraining)

      new ClassifyDataset(
        dataName,
        yScaler.transform(scaledTraining).select("scaledFeatures", "scaledLabel")
          .toDF("features", "label"),
        yScaler.transform(scaledValidation).select("scaledFeatures", "scaledLabel")
          .toDF("features", "label"),
        yScaler.transform(scaledTest).select("scaledFeatures", "scaledLabel")
          .toDF("features", "label"))
    } else {
      new ClassifyDataset(
        dataName,
        scaledTraining.select("scaledFeatures", "label").toDF("features", "label"),
        scaledValidation.select("scaledFeatures", "label").toDF("features", "label"),
        scaledTest.select("scaledFeatures", "label").toDF("features", "label"))
    }
  }
}
