package org.itshared.samsara

import org.apache.mahout.math._
import org.apache.mahout.math.scalabindings._
import org.apache.mahout.math.drm._
import org.apache.mahout.math.scalabindings.RLikeOps._
import org.apache.mahout.math.drm.RLikeDrmOps._
import org.apache.mahout.sparkbindings._

object TestLocal {

  def main(args: Array[String]): Unit = {

    val vec1: Vector = (1.0, 2.0, 3.0)
    val vec2 = dvec(1, 1, 1)
    println(vec1 + vec2)

    val A = dense((1, 2, 3), (3, 4, 5), (5, 6, 7))
    val b: Vector = (1, 2, 5)
    
    // print matrix 
    println(A)
    
    // print row #1 
    println(A(1, ::))
    
    // ranges
    println(A(0 to 1, 1 to 2))

    // solve Ax = b
    println(solve(A, b))

    // or find A^-1
    println(solve(A))
    
    // mult
    println(A.t %*% A)

    // svd
    println(svd(A))
  }

}