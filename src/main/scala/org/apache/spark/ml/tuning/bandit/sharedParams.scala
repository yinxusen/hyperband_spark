package org.apache.spark.ml.tuning.bandit

import org.apache.spark.ml.param.{DoubleParam, IntParam, Params}

/**
 * Created by panda on 8/10/15.
 */
private[ml] trait HasStepControl extends Params {

  /**
   * Param for input column name.
   * @group param
   */
  final val step: IntParam = new IntParam(this, "step", "current step in partial training")
  setDefault(step -> 0)

  /** @group getParam */
  final def getStep: Int = $(step)
}

private[ml] trait HasDownSamplingFactor extends Params {
  final val downSamplingFactor: DoubleParam = new DoubleParam(this, "downSamplingFactor", "down sampling factor")
  setDefault(downSamplingFactor -> 1)
  final def getDownSamplingFactor: Double = $(downSamplingFactor)
}

private[ml] trait HasStepsPerPulling extends Params {
  final val stepsPerPulling: IntParam = new IntParam(this, "stepsPerPulling", "the count of iterative steps in one pulling")
  setDefault(stepsPerPulling -> 1)
  final def getStepsPerPulling: Int = $(stepsPerPulling)
}
