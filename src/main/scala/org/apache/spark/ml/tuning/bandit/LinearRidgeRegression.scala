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

import org.apache.spark.ml.param.shared.{HasFeaturesCol, HasLabelCol, HasOutputCol}
import org.apache.spark.ml.param.{DoubleParam, IntParam, ParamMap, Params}
import org.apache.spark.ml.regression.RegressionModel
import org.apache.spark.ml.util.{Identifiable, SchemaUtils}
import org.apache.spark.mllib.linalg.BLAS._
import org.apache.spark.mllib.linalg.{Vector, VectorUDT, Vectors}
import org.apache.spark.mllib.optimization.{LeastSquaresGradient, SquaredL2Updater}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.{DoubleType, StructType}
import org.apache.spark.sql.{DataFrame, Row}

trait LinearRidgeRegressionParam
  extends Params with HasFeaturesCol with HasLabelCol with HasOutputCol with HasStepControl {

  val numOfFeatures: IntParam = new IntParam(this, "numOfFeatures", "the number of feature columns")
  def getNumOfFeatures: Int = $(numOfFeatures)

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

  def setNumOfFeatures(value: Int): this.type = set(numOfFeatures, value)

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
    val weight = LinearRidgeRegression
      .runSingleStepSGD(
        data, currentStep, Vectors.zeros($(numOfFeatures)), $(step), $(stepsPerPulling))
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
      .runSingleStepSGD(data, currentStep, initModel.weights, $(regularizer), $(stepsPerPulling))
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
  val gradient = new LeastSquaresGradient()
  val updater = new SquaredL2Updater()

  val optimizer = new GradientDescentOptimizer(gradient, updater)

  def runSingleStepSGD(
      data: RDD[LabeledPoint],
      currentStep: Int,
      currentWeight: Vector,
      regularizer: Double,
      steps: Int): Vector = {
    optimizer.setRegParam(regularizer).setCurrentStep(currentStep)
      .optimize(data.map(x => (x.label, x.features)), currentWeight)
  }
}
