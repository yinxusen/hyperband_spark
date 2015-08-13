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

import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.param.ParamMap

import scala.collection.mutable

/**
 * Model family to create arms.
 */
abstract class ModelFamily(val name: String, val paramList: Array[ParamSampler[_]]) {

  /**
   * Create an arm given initial dataset, parameters. The model family provides the
   * [PartialEstimator] plus with necessary settings.
   */
  def createArm(initData: Dataset, params: ParamMap): Arms.ArmExistential

  def addArm(
      hp: ParamMap,
      arms: mutable.Map[(String, String), Arms.ArmExistential],
      arm: Arms.ArmExistential): Unit = {
    arms += (this.name, hp.toString) -> arm
  }

  def createArms(
      hpPoints: Array[ParamMap],
      initData: Dataset,
      arms: mutable.Map[(String, String), Arms.ArmExistential]):
  mutable.Map[(String, String), Arms.ArmExistential] = {
    for (hp <- hpPoints) {
      this.addArm(hp, arms, this.createArm(initData, hp))
    }
    arms
  }
}

/**
 * Linear ridge regression model family, which provides [LinearRidgeRegression] to generate an arm.
 */
class LinearRidgeRegressionModelFamily(
    override val name: String,
    override val paramList: Array[ParamSampler[_]])
  extends ModelFamily(name, paramList) {

  /**
   * Create an arm with a [LinearRidgeRegression] estimator. Note that the params should only
   * contain hyper-parameters. As a consequence, other parameters that will not be changed should
   * be set here.
   *
   * Note that the names of feature column, label column, prediction column in
   * [LinearRidgeRegression], [LinearRidgeRegressionModel], and [Evaluator] should be the same.
   */
  override def createArm(initData: Dataset, params: ParamMap): Arm[LinearRidgeRegressionModel] = {
    val linearRidgeRegression= new LinearRidgeRegression()
      .setDownSamplingFactor(0.1).setStepsPerPulling(1).copy(params)

    val evaluator = new RegressionEvaluator().setMetricName("rmse")

    new Arm[LinearRidgeRegressionModel](
      initData, "linear ridge regression", linearRidgeRegression, evaluator)
  }
}