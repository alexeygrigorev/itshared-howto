package org.itshared.samsara

import org.apache.mahout.math._
import org.apache.mahout.math.scalabindings._
import org.apache.mahout.math.drm._
import org.apache.mahout.math.scalabindings.RLikeOps._
import org.apache.mahout.math.drm.RLikeDrmOps._
import org.apache.mahout.sparkbindings._
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf

/**
 * Tutorial from https://github.com/sscdotopen/krams/blob/master/linear-regression-cereals.md
 */
object CerealRegression {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf()
      .setAppName("Cereals OLS Regression")
      .setMaster("local[*]")
      .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .set("spark.kryo.registrator", "org.apache.mahout.sparkbindings.io.MahoutKryoRegistrator")
    implicit val sc = new SparkDistributedContext(new SparkContext(conf))

    val matrix = dense(
      (2, 2, 10.5, 10, 29.509541), // Apple Cinnamon Cheerios
      (1, 2, 12, 12, 18.042851), // Cap'n'Crunch
      (1, 1, 12, 13, 22.736446), // Cocoa Puffs
      (2, 1, 11, 13, 32.207582), // Froot Loops
      (1, 2, 12, 11, 21.871292), // Honey Graham Ohs
      (2, 1, 16, 8, 36.187559), // Wheaties Honey Gold
      (6, 2, 17, 1, 50.764999), // Cheerios
      (3, 2, 13, 7, 40.400208), // Clusters
      (3, 3, 13, 4, 45.811716)) // Great Grains Pecan)

    val drmData = drmParallelize(matrix, numPartitions = 2)

    val drmX = drmData(::, 0 until 4)
    val y = drmData.collect(::, 4)

    val drmXtX = drmX.t %*% drmX
    val drmXty = drmX.t %*% y

    val XtX = drmXtX.collect
    val Xty = drmXty.collect(::, 0)

    println(XtX)

    val beta = solve(XtX, Xty)
    println(XtX)

    val yFitted = (drmX %*% beta).collect(::, 0)
    println(yFitted)

    val sol = ols(drmX, y)
    println(goodnessOfFit(drmX, sol, y))

  }

  def ols(drmX: DrmLike[Int], y: Vector) = {
    val sol = solve(drmX.t %*% drmX, drmX.t %*% y)
    sol(::, 0)
  }

  def goodnessOfFit(drmX: DrmLike[Int], beta: Vector, y: Vector) = {
    val fittedY = (drmX %*% beta) collect (::, 0)
    (y - fittedY).norm(2)
  }

}