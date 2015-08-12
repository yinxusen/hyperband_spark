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

import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.sql.DataFrame

/**
 * Partial estimator performs a single iterative step in each fitting.
 */
abstract class PartialEstimator[M <: Model[M]]
  extends Estimator[M] with HasDownSamplingFactor with HasStepsPerPulling {

  def fit(dataset: DataFrame, initModel: M): M

  def fit(dataset: DataFrame, initModel: M, paramMap: ParamMap): M = {
    copy(paramMap).fit(dataset, initModel)
  }

  def fit(dataset: DataFrame, initModel: M, paramMaps: Array[ParamMap]): Seq[M] = {
    paramMaps.map(fit(dataset, initModel, _))
  }

  override def copy(extra: ParamMap): PartialEstimator[M]
}
