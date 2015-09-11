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

import org.apache.spark.rdd.{PartitionwiseSampledRDD, RDD}
import org.apache.spark.util.random.BernoulliCellSampler

import scala.reflect.ClassTag

object Utils {
  def splitTrainTest[T: ClassTag](rdd: RDD[T], testFraction: Double, seed: Int): (RDD[T], RDD[T]) = {
    val sampler = new BernoulliCellSampler[T](0, testFraction, complement = false)
    val test = new PartitionwiseSampledRDD(rdd, sampler, true, seed)
    val training = new PartitionwiseSampledRDD(rdd, sampler.cloneComplement(), true, seed)
    (training, test)
  }
}
