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

import org.apache.spark.ml.attribute.AttributeGroup
import org.apache.spark.ml.param.shared.{HasLabelCol, HasFeaturesCol, HasInputCol, HasOutputCol}
import org.apache.spark.ml.param.{DoubleParam, ParamMap, Params}
import org.apache.spark.ml.regression.RegressionModel
import org.apache.spark.ml.util.{Identifiable, SchemaUtils}
import org.apache.spark.mllib.linalg.BLAS._
import org.apache.spark.mllib.linalg.{Vector, VectorUDT}
import org.apache.spark.mllib.regression.{LabeledPoint, RidgeRegressionWithSGD}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.sql.types.{DoubleType, StructType}

trait LinearRidgeRegressionParam
  extends Params with HasFeaturesCol with HasLabelCol with HasOutputCol with HasStepControl {

  /**
   * Regularization parameter for linear ridge regression.
   *
   * @group param
   */
  val regularizer: DoubleParam = new DoubleParam(this, "regularizer", "regularization parameter")
  setDefault(regularizer -> 0.1)

  /** @group getParam */
  def getRegularizer: Double = $(regularizer)

  /**
   * Validate and transform the input schema.
   */
  protected def validateAndTransformSchema(schema: StructType): StructType = {
    SchemaUtils.checkColumnType(schema, $(featuresCol), new VectorUDT)
    SchemaUtils.checkColumnType(schema, $(labelCol), DoubleType)
    schema
  }
}

class LinearRidgeRegression(override val uid: String)
  extends PartialEstimator[LinearRidgeRegressionModel] with LinearRidgeRegressionParam {

  def this() = this(Identifiable.randomUID("linear ridge regression"))

  /** @group setParam */
  def setRegularizer(value: Double): this.type = set(regularizer, value)

  /** @group setParam */
  def setStep(value: Int): this.type = set(step, value)

  /** @group setParam */
  def setStepsPerPulling(value: Int): this.type  = set(stepsPerPulling, value)

  /** @group setParam */
  def setDownSamplingFactor(value: Double): this.type = set(downSamplingFactor, value)

  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema)
  }

  /**
   * Fit the [LinearRidgeRegressionModel] in a single SGD step without an initial model. In this
   * condition, the code will generate a initial model itself and usually, it will be a all-zero
   * model.
   */
  override def fit(dataset: DataFrame): LinearRidgeRegressionModel = {
    val currentStep = $(step) + 1
    this.setStep(currentStep)
    val data = dataset.map { case Row(x: Vector, y: Double) => LabeledPoint(y, x)}
    val weight =
      LinearRidgeRegression.singleSGDStep(data, currentStep, None, $(step), $(stepsPerPulling))
    new LinearRidgeRegressionModel(uid, weight, 0)
  }

  /**
   * Fit a [LinearRidgeRegressionModel] in a single SGD step with an initial model.
   */
  override def fit(
      dataset: DataFrame,
      initModel: LinearRidgeRegressionModel): LinearRidgeRegressionModel = {
    val currentStep = $(step) + 1
    this.setStep(currentStep)
    val data = dataset.map { case Row(x: Vector, y: Double) => LabeledPoint(y, x)}
    val weight = LinearRidgeRegression
      .singleSGDStep(data, currentStep, Some(initModel.weights), $(step), $(stepsPerPulling))
    new LinearRidgeRegressionModel(uid, weight, initModel.intercept)
  }

  override def copy(extra: ParamMap): LinearRidgeRegression = defaultCopy(extra)

}

class LinearRidgeRegressionModel(
    override val uid: String,
    val weights: Vector,
    val intercept: Double)
  extends RegressionModel[Vector, LinearRidgeRegressionModel]
  with LinearRidgeRegressionParam {

  override protected def predict(features: Vector): Double = {
    dot(features, weights) + intercept
  }

  override def copy(extra: ParamMap): LinearRidgeRegressionModel = {
    copyValues(new LinearRidgeRegressionModel(uid, weights, intercept), extra)
  }
}

object LinearRidgeRegression {
  def singleSGDStep(
      data: RDD[LabeledPoint],
      currentStep: Int,
      currentWeight: Option[Vector],
      regularizer: Double,
      steps: Int): Vector = {
    val eta = 0.01 / math.sqrt(2.0 + currentStep)

    val stepSize = 0.01 / math.sqrt(2 + currentStep)
    val newModel = if (currentWeight == None) {
      RidgeRegressionWithSGD.train(data, steps, stepSize, regularizer, 0.1)
    } else {
      RidgeRegressionWithSGD.train(data, steps, stepSize, regularizer, 0.1, currentWeight.get)
    }
    newModel.weights
  }
}
