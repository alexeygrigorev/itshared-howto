package org.itshared.samsara

import org.apache.mahout.math._
import org.apache.mahout.math.scalabindings._
import org.apache.mahout.math.drm._
import org.apache.mahout.math.scalabindings.RLikeOps._
import org.apache.mahout.math.drm.RLikeDrmOps._
import org.apache.mahout.sparkbindings._

import org.apache.spark.SparkContext
import org.apache.spark.SparkConf

object TestSpark {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf()
      .setAppName("Simple Application")
      .setMaster("local[*]")
      .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .set("spark.kryo.registrator", "org.apache.mahout.sparkbindings.io.MahoutKryoRegistrator")
    implicit val sc = new SparkDistributedContext(new SparkContext(conf))

    // System.setProperty("mahout.home", "c:/mahout/")
//     implicit val ctx = mahoutSparkContext(masterUrl="local[*]", appName="mahout spark binding")

    val inCoreA = dense((1, 2, 3), (3, 4, 5), (5, 6, 7))
    val A = drmParallelize(inCoreA)
    println(A.t %*% A)

  }

}


