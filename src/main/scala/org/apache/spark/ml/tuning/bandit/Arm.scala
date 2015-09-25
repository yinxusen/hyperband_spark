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

import org.apache.spark.ml.Model
import org.apache.spark.ml.evaluation.Evaluator
import org.apache.spark.ml.param.{DoubleParam, IntParam, ParamMap}

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

/**
 * Keep record of compressed histories in an iterative program.
 *
 * @param iterations The iteration number as the X-axis.
 * @param errors The error at each iteration as the Y-axis.
 * @param alpha The decay parameter so as to compress the errors.
 */
class CompressedHistory(
    var doCompute: Boolean,
    var iterations: Array[Int],
    var errors: Array[Double],
    var alpha: Double)

/**
 * Multi-bandit arm for hyper-parameter selection. An arm is a composition of an estimator, a model
 * and an evaluator. Pulling an arm means performs a single iterative step for the estimator, which
 * consumes a current model and produce a new one. The evaluator computes the error given a target
 * column and a predicted column.
 */
class Arm[M <: Model[M]](
    var data: Dataset,
    val name: String,
    val estimator: PartialEstimator[M],
    val evaluator: Evaluator) {

  var model: Option[M] = None

  var numPulls: Int = 0
  var numEvals: Int = 0

  var results: Array[Double] = Array.empty

  val history: CompressedHistory = new CompressedHistory(false, Array.empty, Array.empty, 1.2)

  /**
   * Reset the arm to use in a new [SearchStrategy].
   */
  def reset(): this.type = {
    this.results = Array.empty
    this.numPulls = 0
    this.numEvals = 0
    this.model = None
    this
  }

  /**
   * Strip the arm so as to save the storage space. Only the key items are retained.
   */
  def strip(): this.type = {
    this.data = null
    this.model = None
    this.history.iterations = Array.empty
    this.history.errors = Array.empty
    this
  }

  /**
   * Pull the arm to perform one step of the iterative [PartialEstimator]. Model will be updated
   * after the pulling.
   */
  def pull(): this.type = {
    this.numPulls += 1
    val partialModel = if (model == None) {
      this.estimator.fit(data.trainingSet)
    } else {
      this.estimator.fit(data.trainingSet, model.get)
    }
    this.model = Some(partialModel)
    println(s"Arm ${this.name}, iteration ${this.numPulls}, validation ${this.getValidationResult()}")
    this
  }

  /**
   * Train the [PartialEstimator] in `maxIter` times, so as to get a ultimate model. Model will be
   * updated after training. The (iteration, error) pair of training process will be kept in the
   * `abridgedHistory`.
   */
  def train(maxIter: Int): this.type = {
    this.reset()
    val valXArrayBuffer = new ArrayBuffer[Int]()
    val valYArrayBuffer = new ArrayBuffer[Double]()
    while (this.numPulls < maxIter) {
      this.pull()
      if (this.history.doCompute) {
        if (this.history.iterations.size == 0 ||
          this.numPulls > valXArrayBuffer.last * this.history.alpha) {
          valXArrayBuffer.append(this.numPulls)
          val error = this.getValidationResult()
          valYArrayBuffer.append(error)
        }
      }
    }
    this.history.iterations = valXArrayBuffer.toArray
    this.history.errors = valYArrayBuffer.toArray
    this
  }

  /**
   * Evaluate the model according to training, validation, and test set.
   *
   * @param part "train", "validation", or "test". Specify each of them then evaluating it.
   * @return An array of (training set error, validation set error, test set error).
   */
  def getResults(recompute: Boolean = true, part: Option[String] = None): Array[Double] = {
    if (model.isEmpty) {
      throw new Exception("model is empty")
    } else {
      if (this.results.isEmpty || recompute) {
        this.numEvals += 1
        if (part.isEmpty || this.results.isEmpty) {
          this.results = Array(evaluator.evaluate(model.get.transform(data.trainingSet)),
            evaluator.evaluate(model.get.transform(data.validationSet)),
            evaluator.evaluate(model.get.transform(data.testSet)))
        } else if (part == Some("train")) {
          this.results(0) = evaluator.evaluate(model.get.transform(data.trainingSet))
        } else if (part == Some("validation")) {
          this.results(1) = evaluator.evaluate(model.get.transform(data.validationSet))
        } else if (part == Some("test")) {
          this.results(2) = evaluator.evaluate(model.get.transform(data.testSet))
        } else {
          throw new Exception(
            "Parameter error! You can only select from [\"train\", \"validation\", \"test\"].")
        }
      }
    }
    this.results
  }

  /**
   * Evaluate and get the training result.
   */
  def getTrainResult(recompute: Boolean = true): Double = getResults(recompute, Some("train"))(0)

  /**
   * Evaluate and get the validation result.
   */
  def getValidationResult(recompute: Boolean = true): Double =
    getResults(recompute, Some("validation"))(1)

  /**
   * Evaluate and get the test result.
   */
  def getTestResult(recompute: Boolean = true): Double = getResults(recompute, Some("test"))(2)
}

object Arms {
  type ArmExistential = Arm[M] forSome {type M <: Model[M]}

  /**
   * Generate arms given an array of [ModelFamily].
   *
   * @param numArmsPerParameter Number of sampling parameters sampled from [ParamSampler].
   * @return Map of (model family name, hyper-parameter name) -> arm.
   */
  def generateArms(
      armFactories: Array[ArmFactory],
      data: Dataset,
      numArmsPerParameter: Int): Map[(String, String), Arms.ArmExistential] = {
    val arms = new mutable.HashMap[(String, String), Arms.ArmExistential]()
    for (armFactory <- armFactories) {
      val numParamsToTune = armFactory.paramList.size
      val numArmsForModelFamily = numParamsToTune * numArmsPerParameter
      val hyperParameterPoints = (0 until numArmsForModelFamily).map { index =>
        val paramMap = new ParamMap()
        armFactory.paramList.map {
          case parameter@(_: IntParamSampler) =>
            val param = new IntParam("from arm", parameter.name, "arm generated parameter")
            paramMap.put(param, parameter.getOneRandomSample)
          case parameter@(_: DoubleParamSampler) =>
            val param = new DoubleParam("from arm", parameter.name, "arm generated parameter")
            paramMap.put(param, parameter.getOneRandomSample)
          case _ => throw new UnsupportedOperationException("Not implemented yet.")
        }
        paramMap
      }.toArray
      armFactory.createArms(hyperParameterPoints, data, arms)
    }
    arms.toMap
  }
}
