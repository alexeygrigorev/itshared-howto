package org.itshared.samsara

import org.apache.mahout.math._
import org.apache.mahout.math.scalabindings._
import org.apache.mahout.math.drm._
import org.apache.mahout.math.scalabindings.RLikeOps._
import org.apache.mahout.math.drm.RLikeDrmOps._
import org.apache.spark.SparkConf
import org.apache.mahout.sparkbindings._

import org.apache.spark.SparkContext
import collection._
import JavaConversions._

import org.apache.mahout.math.decompositions.DQR._
import org.apache.mahout.math.decompositions.DSPCA._
import org.apache.mahout.math.decompositions.DSSVD._

object YearPredictionMSD {

  def main(args: Array[String]) {
    val textFilePath = "file:///c:/tmp/203/YearPredictionMSD.txt"
    val numPartitions = 2

    val conf = new SparkConf()
      .setAppName("Cereals OLS Regression")
      .setMaster("local[*]")
      .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .set("spark.kryo.registrator", "org.apache.mahout.sparkbindings.io.MahoutKryoRegistrator")
    implicit val sc = new SparkDistributedContext(new SparkContext(conf))

    val data = sc.textFile(textFilePath, numPartitions)
    val rdd = data.map(line => line.split(",").map(_.toDouble)).map(dvec(_))
    val rddMatrixLike: DrmRdd[Int] = rdd.zipWithIndex.map { case (v, idx) => (idx.toInt, v) }

    val matrix = drmWrap(rddMatrixLike)
    val nrow = matrix.nrow.toInt
    val ncol = matrix.ncol

    val train = matrix(0 until 463715, ::)
    val test = matrix(463715 until nrow, ::)

    val X = train(::, 1 until ncol)
    val y = train.viewColumn(0)

    val w = qr_ols(add_bias(X), y)

    val X_test = test(::, 1 until ncol)
    val y_test = test.viewColumn(0)

    val error = goodnessOfFit(add_bias(X_test), w, y_test)
    println("error ols", error)

    val X_std = add_bias(standardize(X))
    val X_test_std = add_bias(standardize(X_test))
    val ridgeFunc = ridge(X_std, y)

    var best_lambda = 0.0
    var best_fit = Double.MaxValue

    for (lambda <- Seq(0.1, 0.5, 1.0, 3.0, 5.0, 10.0)) {
      val w_reg = ridgeFunc(lambda)
      val error_reg = goodnessOfFit(X_test_std, w_reg, y_test)
      println("error ridge", s"lambda = $lambda", error_reg)
      if (error_reg < best_fit) {
        best_fit = error_reg
        best_lambda = lambda
      }
    }

    println("best fit ridge", s"lambda = $best_lambda", best_fit)
  }


  def add_bias(drmX: DrmLike[Int]) = {
    drmX.mapBlock(drmX.ncol + 1) {
      case (keys, block) => {
        val block_new = block.like(block.nrow, block.ncol + 1)
        block_new.zip(block).foreach {
          case (row_new, row_orig) =>
            row_new(0) = 1.0
            row_new(1 to block.ncol) := row_orig
        }
        keys -> block_new
      }
    }
  }

  def standardize(drmX: DrmLike[Int])(implicit ctx: DistributedContext) = {
    val meanVec = drmX.colMeans
    val variance = (drmX * drmX).colMeans - (meanVec * meanVec)

    val mean = drmBroadcast(meanVec)
    val std = drmBroadcast(variance.sqrt)

    val res = drmX.mapBlock(drmX.ncol) {
      case (keys, block) => {
        val copy = block.cloned
        copy.foreach(row => row := (row - mean) / std)
        (keys, copy)
      }
    }

    res
  }

  def ols(drmX: DrmLike[Int], y: Vector) = {
    val sol = solve(drmX.t %*% drmX, drmX.t %*% y)
    sol(::, 0)
  }

  def qr_ols(drmX: DrmLike[Int], y: Vector) = {
    val QR = dqrThin(drmX)
    val Q = QR._1
    val R = QR._2

    val sol = solve(R, Q.t %*% y)
    sol(::, 0)
  }

  def ridge(drmX: DrmLike[Int], y: Vector): (Double) => Vector = {
    val XTX = drmX.t %*% drmX
    val XTy = drmX.t %*% y

    (lambda: Double) => {
      val reg = diag(lambda, XTX.ncol)
      val sol = solve(XTX.plus(reg), XTy)
      sol(::, 0)
    }
  }

  def goodnessOfFit(drmX: DrmLike[Int], w: Vector, y: Vector) = {
    val fittedY = (drmX %*% w) collect (::, 0)
    (y - fittedY).norm(2)
  }

}