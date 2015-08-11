package org.apache.spark.ml.tuning.bandit

import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.sql.DataFrame

/**
 * Created by panda on 8/2/15.
 */
abstract class PartialEstimator[M <: Model[M]] extends Estimator[M] with HasDownSamplingFactor with HasStepsPerPulling {

  def setStepsPerPulling(value: Int): this.type  = set(stepsPerPulling, value)

  def setDownSamplingFactor(value: Double): this.type = set(downSamplingFactor, value)

  def fit(dataset: DataFrame, initModel: M, paramMap: ParamMap): M = {
    copy(paramMap).fit(dataset, initModel)
  }

  def fit(dataset: DataFrame, initModel: M): M

  def fit(dataset: DataFrame, initModel: M, paramMaps: Array[ParamMap]): Seq[M] = {
    paramMaps.map(fit(dataset, initModel, _))
  }

  override def copy(extra: ParamMap): PartialEstimator[M]
}
