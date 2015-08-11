package org.apache.spark.ml.tuning.bandit

import org.apache.spark.rdd.{PartitionwiseSampledRDD, RDD}
import org.apache.spark.util.random.BernoulliCellSampler

import scala.reflect.ClassTag

/**
 * Created by panda on 8/9/15.
 */
object Utils {
  def splitTrainTest[T: ClassTag](rdd: RDD[T], testSize: Double, seed: Int): (RDD[T], RDD[T]) = {
    val sampler = new BernoulliCellSampler[T](0, testSize, complement = false)
    val test = new PartitionwiseSampledRDD(rdd, sampler, true, seed)
    val training = new PartitionwiseSampledRDD(rdd, sampler.cloneComplement(), true, seed)
    (training, test)
  }
}
