/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.ml.tuning.bandit

import org.apache.spark.ml.feature.StandardScaler
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.{DataFrame, Row, SQLContext}

abstract class Dataset(
    val name: String,
    val trainingSet: DataFrame,
    val validationSet: DataFrame,
    val testSet: DataFrame,
    val numOfFeatures: Int) {

  /**
   * A base validation error.
   */
  def baseValErr(): Double
}

/**
 * Classification dataset contains training, validation, and test set. Each of them is a
 * [DataFrame], of which the `label` column represents a `Double` target, while the `features`
 * column is refer to features, which is a `Vector`.
 *
 * Note that column names cannot modify, in accord to other processing scripts.
 */
class ClassifyDataset(
    override val name: String,
    override val trainingSet: DataFrame,
    override val validationSet: DataFrame,
    override val testSet: DataFrame,
    override val numOfFeatures: Int)
  extends Dataset(name, trainingSet, validationSet, testSet, numOfFeatures) {

  override def baseValErr(): Double = {
    val ySum = validationSet.select("label").map { case Row(x: Double) => x}.sum()
    val xCnt = validationSet.count()
    0.5 - math.abs(ySum) / (2 * xCnt)
  }
}

object ClassifyDataset {
  /**
   * Partitioning the given data (via file name) into training, validation, and test set, then
   * standard scaling them according to the training set.
   *
   * All features would be scaled, however the scaling of label is optional.
   */
  def scaleAndPartitionData(
      sqlCtx: SQLContext,
      dataName: String,
      fileName: String,
      normalizeY: Boolean = false): ClassifyDataset = {
    val sc = sqlCtx.sparkContext
    val data = MLUtils.loadLibSVMFile(sc, fileName).map { case LabeledPoint(label, features) =>
      LabeledPoint(label, features.toDense)
    }

    val numOfFeatures = data.first().features.size

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
          .toDF("features", "label").cache(),
        yScaler.transform(scaledValidation).select("scaledFeatures", "scaledLabel")
          .toDF("features", "label").cache(),
        yScaler.transform(scaledTest).select("scaledFeatures", "scaledLabel")
          .toDF("features", "label").cache(),
        numOfFeatures)
    } else {
      new ClassifyDataset(
        dataName,
        scaledTraining.select("scaledFeatures", "label").toDF("features", "label").cache(),
        scaledValidation.select("scaledFeatures", "label").toDF("features", "label").cache(),
        scaledTest.select("scaledFeatures", "label").toDF("features", "label").cache(),
        numOfFeatures)
    }
  }
}
