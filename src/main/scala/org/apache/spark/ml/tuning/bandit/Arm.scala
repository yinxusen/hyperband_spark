package org.apache.spark.ml.tuning.bandit

import org.apache.spark.ml.Model
import org.apache.spark.ml.evaluation.Evaluator
import org.apache.spark.ml.param.{DoubleParam, IntParam, ParamMap}

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

/**
 * Created by panda on 7/31/15.
 */

class AbridgedHistory(var compute: Boolean, var iterations: Array[Int], var errors: Array[Double], var alpha: Double)

class Arm[M <: Model[M]](
    var data: Dataset,
    var model: Option[M],
    var numPulls: Int,
    var numEvals: Int,
    val modelType: String,
    val estimator: PartialEstimator[M],
    val evaluator: Evaluator,
    var results: Array[Double] = Array.empty,
    val abridgedHistory: AbridgedHistory = new AbridgedHistory(false, Array.empty, Array.empty, 1.2)) {

  def reset(): this.type = {
    this.results = Array.empty
    this.numPulls = 0
    this.numEvals = 0
    this.model = None
    this
  }

  def stripArm(): Unit = {
    this.data = null
    this.model = None
    this.abridgedHistory.iterations = Array.empty
    this.abridgedHistory.errors = Array.empty
  }

  def pullArm(): Unit = {
    this.numPulls += 1
    val partialModel = if (model == None) {
      this.estimator.fit(data.trainingSet)
    } else {
      this.estimator.fit(data.trainingSet, model.get)
    }
    this.model = Some(partialModel)
  }

  def trainToCompletion(maxIter: Double): Unit = {
    this.reset()
    val valXArrayBuffer = new ArrayBuffer[Int]()
    val valYArrayBuffer = new ArrayBuffer[Double]()
    while (this.numPulls < maxIter) {
      this.pullArm()
      if (this.abridgedHistory.compute) {
        if (this.abridgedHistory.iterations.size == 0 || this.numPulls > valXArrayBuffer.last * this.abridgedHistory.alpha) {
          valXArrayBuffer.append(this.numPulls)
          val error = this.getResults(true, Some("validation"))(1)
          valYArrayBuffer.append(error)
        }
      }
    }
    this.abridgedHistory.iterations = valXArrayBuffer.toArray
    this.abridgedHistory.errors = valYArrayBuffer.toArray
  }

  def getResults(forceRecompute: Boolean = true, partition: Option[String] = None): Array[Double] = {
    if (model.isEmpty) {
      throw new Exception("model is empty")
    } else {
      if (this.results == Array.empty || forceRecompute) {
        this.numEvals += 1
        if (partition == None || this.results == Array.empty) {
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
          // TODO
        }
      }
    }
    this.results
  }
}

object Arms {
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
          case _ =>
          // TODO refine the code
        }
        paramMap
      }.toArray
      modelFamily.createArms(hyperParameterPoints, data, arms)
    }
    arms.toMap
  }
}

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
