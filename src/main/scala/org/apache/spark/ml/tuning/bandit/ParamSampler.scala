package org.apache.spark.ml.tuning.bandit

import scala.util.Random

/**
 * Created by panda on 8/10/15.
 */
abstract class ParamSampler[T](val name: String, val paramType: String) {
  def getSamples(numVal: Int): Array[T]

  def getOneRandomSample: T
}

class IntParamSampler(
    override val name: String,
    val minVal: Int,
    val maxVal: Int,
    val initVal: Int = None) extends ParamSampler[Int](name, _) {

  override val paramType: String = "integer"

  override def getSamples(numVal: Int): Array[Int] = {
    ParamSampler.evenlySamplingFromRange(minVal.toDouble, maxVal.toDouble, numVal).map(_.toInt)
  }

  override def getOneRandomSample: Int = {
    ParamSampler.samplingOneFromRange(minVal.toDouble, maxVal.toDouble).toInt
  }
}

class DoubleParamSampler(
    override val name: String,
    val minVal: Double,
    val maxVal: Double,
    val scale: String = "log",
    val initVal: Double = None) extends ParamSampler[Double](name, _) {

  override val paramType: String = "continuous"

  override def getSamples(numVal: Int): Array[Double] = {
    if (this.scale == "log") {
      val minExponent = math.log10(minVal)
      val maxExponent = math.log10(maxVal)
      ParamSampler.evenlySamplingFromRange(minExponent, maxExponent, numVal)
        .map(x => math.pow(10, x))
    } else {
      ParamSampler.evenlySamplingFromRange(minVal, maxVal, numVal)
    }
  }

  override def getOneRandomSample: Double = {
    if (this.scale == "log") {
      val minExponent = math.log10(minVal)
      val maxExponent = math.log10(maxVal)
      math.pow(10, ParamSampler.samplingOneFromRange(minExponent, maxExponent))
    } else {
      ParamSampler.samplingOneFromRange(minVal, maxVal)
    }
  }
}

object ParamSampler {
  def evenlySamplingFromRange(minVal: Double, maxVal: Double, numVal: Int): Array[Double] = {
    val stepSize = (maxVal - minVal) * 1.0 / numVal
    val result = Array.fill[Double](numVal)(0)
    var i = 0
    while (i < numVal) {
      result(i) = minVal + i * stepSize
      i += 1
    }
    result
  }

  def samplingOneFromRange(minVal: Double, maxVal: Double): Double = {
    Random.nextDouble() * (1 + maxVal - minVal) + minVal
  }
}
