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
 * Keep record of abridged histories in an iterative program.
 *
 * @param iterations The iteration number as the X-axis.
 * @param errors The error at each iteration as the Y-axis.
 * @param alpha The decay parameter so as to abridge the errors.
 */
class AbridgedHistory(
    var compute: Boolean,
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
    val modelType: String,
    val estimator: PartialEstimator[M],
    val evaluator: Evaluator) {

  var model: Option[M] = None

  var numPulls: Int = 0
  var numEvals: Int = 0

  var results: Array[Double] = Array.empty

  val abridgedHistory: AbridgedHistory = new AbridgedHistory(false, Array.empty, Array.empty, 1.2)

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
  def stripArm(): Unit = {
    this.data = null
    this.model = None
    this.abridgedHistory.iterations = Array.empty
    this.abridgedHistory.errors = Array.empty
  }

  /**
   * Pull the arm to perform one step of the iterative [PartialEstimator]. Model will be updated
   * after the pulling.
   */
  def pullArm(): Unit = {
    this.numPulls += 1
    val partialModel = if (model == None) {
      this.estimator.fit(data.trainingSet)
    } else {
      this.estimator.fit(data.trainingSet, model.get)
    }
    this.model = Some(partialModel)
  }

  /**
   * Train the [PartialEstimator] in `maxIter` times, so as to get a ultimate model. Model will be
   * updated after training. The (iteration, error) pair of training process will be kept in the
   * `abridgedHistory`.
   */
  def trainToCompletion(maxIter: Double): Unit = {
    this.reset()
    val valXArrayBuffer = new ArrayBuffer[Int]()
    val valYArrayBuffer = new ArrayBuffer[Double]()
    while (this.numPulls < maxIter) {
      this.pullArm()
      if (this.abridgedHistory.compute) {
        if (this.abridgedHistory.iterations.size == 0 ||
          this.numPulls > valXArrayBuffer.last * this.abridgedHistory.alpha) {
          valXArrayBuffer.append(this.numPulls)
          val error = this.getResults(true, Some("validation"))(1)
          valYArrayBuffer.append(error)
        }
      }
    }
    this.abridgedHistory.iterations = valXArrayBuffer.toArray
    this.abridgedHistory.errors = valYArrayBuffer.toArray
  }

  /**
   * Evaluate the model according to training, validation, and test set.
   *
   * @param partition "train", "validation", or "test". Specify each of them then evaluating it.
   * @return An array of (training set error, validation set error, test set error).
   */
  def getResults(
      forceRecompute: Boolean = true, partition: Option[String] = None): Array[Double] = {
    if (model.isEmpty) {
      throw new Exception("model is empty")
    } else {
      if (this.results.isEmpty || forceRecompute) {
        this.numEvals += 1
        if (partition.isEmpty || this.results.isEmpty) {
          this.results = Array(evaluator.evaluate(model.get.transform(data.trainingSet)),
            evaluator.evaluate(model.get.transform(data.validationSet)),
            evaluator.evaluate(model.get.transform(data.testSet)))
        } else if (partition == Some("train")) {
          this.results(0) = evaluator.evaluate(model.get.transform(data.trainingSet))
        } else if (partition == Some("validation")) {
          this.results(1) = evaluator.evaluate(model.get.transform(data.validationSet))
        } else if (partition == Some("test")) {
          this.results(2) = evaluator.evaluate(model.get.transform(data.testSet))
        } else {
          throw new Exception(
            "Parameter error! You can only select from [\"train\", \"validation\", \"test\"].")
        }
      }
    }
    this.results
  }
}

object Arms {

  /**
   * Generate arms given an array of [ModelFamily].
   *
   * @param numArmsPerParameter Number of sampling parameters sampled from [ParamSampler].
   * @return Map of (model family name, hyper-parameter name) -> arm.
   */
  def generateArms(
      modelFamilies: Array[ModelFamily],
      data: Dataset,
      numArmsPerParameter: Int): Map[(String, String), Arm] = {
    val arms = new mutable.HashMap[(String, String), Arm]()
    for (modelFamily <- modelFamilies) {
      val numParamsToTune = modelFamily.paramList.size
      val numArmsForModelFamily = numParamsToTune * numArmsPerParameter
      val hyperParameterPoints = (0 until numArmsForModelFamily).map { index =>
        val paramMap = new ParamMap()
        modelFamily.paramList.map {
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
      modelFamily.createArms(hyperParameterPoints, data, arms)
    }
    arms.toMap
  }
}

/**
 * Allocate an array of pre-generated arms for a [SearchStrategy].
 */
class ArmsAllocator(val allArms: Map[(String, String), Arm]) {
  val usedArms = new ArrayBuffer[(String, String)]()
  val unusedArms = new ArrayBuffer[(String, String)]()
  unusedArms.appendAll(allArms.keys)
  val arms = new mutable.HashMap[(String, String), Arm]()

  def allocate(numArms: Int): Map[(String, String), Arm] = {
    assert(numArms <= allArms.size, "Required arms exceed the total amount.")
    val arms = new mutable.HashMap[(String, String), Arm]()
    var i = 0
    while (i < math.min(numArms, usedArms.size)) {
      arms += usedArms(i) -> allArms(usedArms(i))
      i += 1
    }
    while (i < numArms) {
      val armInfo = unusedArms.remove(0)
      arms += armInfo -> allArms(armInfo)
      usedArms.append(armInfo)
      i += 1
    }
    arms.toMap.mapValues(_.reset())
  }
}
