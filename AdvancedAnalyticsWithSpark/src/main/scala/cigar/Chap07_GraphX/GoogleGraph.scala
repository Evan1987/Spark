package cigar.Chap07_GraphX

import cigar.Constant._
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.SparkSession
import org.apache.spark.graphx._

object GoogleGraph {
  val folderPath = constant.PATH + "ch07-graph\\test_data\\"
  val dataPath = folderPath + "test2.txt"

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    val spark = SparkSession.builder().appName("TestGraph").master("local[4]").getOrCreate()
    val graph = GraphLoader.edgeListFile(spark.sparkContext, dataPath, canonicalOrientation=true)



    /*val sourceId: VertexId = 3
    val sssp = getSingleSourceShortestPathLength(graph, sourceId)
    sssp.vertices.collect().foreach{case (targetID, dist) => println(s"$sourceId -> $targetID:  $dist")}*/

    //**** test1
    /*val testResult = exploreMsgSend(graph)
    testResult.vertices.collect().sortBy(_._1).foreach{case (id, value) => println(s"$id: $value")}*/

    //**** test2
    /*val testResult = getBiggestNum(graph)
    testResult.vertices.collect().sortBy(_._1).foreach{case (id, value) => println(s"$id: $value")}*/

    //**** test3
    val testResult = exploreMsgAgg(graph)
    testResult.collect().sortBy(_._1).foreach{case (id, value) => println(s"$id: $value")}
  }

  /**
    * *************** 计算原点距离各节点的最短路径长度 *******************
    * */
  def getSingleSourceShortestPathLength[V, E](graph: Graph[V, E], sourceId: VertexId): Graph[Double, Int] = {

    // 初始化顶点属性（距离原点的最短路径长度）：原点自身长度为0，其余均为正无穷
    // 默认边属性为 int 1
    val initialGraph: Graph[Double, Int] = graph
      .mapVertices((id, _) => if (id == sourceId) 0.0 else Double.PositiveInfinity)
      .mapTriplets(triplet => 1)

    // 初始化消息，定义超步0时，每个节点接收到的消息
    val initialMessage = Double.PositiveInfinity
    // 节点计算函数 Vertex Program， 定义节点上接收到消息后如何计算
    val vprog = (id: VertexId, dist: Double, newDist: Double) => math.min(dist, newDist)

    // 消息传递函数，定义节点如何向外传递消息，是否发消息
    // 消息的发送节点范围一定在该triplet的两个端点内，不允许跨边
    val sendMsg = (triplet: EdgeTriplet[Double, Int]) =>
      // 第一次执行后，只有与原点直接相连的节点是活跃的，其余都是不活跃的
      if (triplet.srcAttr + triplet.attr < triplet.dstAttr) Iterator((triplet.srcId, triplet.srcAttr + triplet.attr)) else Iterator.empty
    // 消息聚合函数，当目标节点收到多方消息时如何聚合消息
    val mergeMsg = (a: Double, b: Double) => math.min(a, b)

    // EdgeDirection说明
    // In -> 边的dstID必须是活跃的，才会触发消息传递
    // Out -> 边的srcID必须是活跃的，才会触发消息传递
    // Both -> 边的srcID和dstID必须都是活跃的，才会触发消息传递
    // Either -> 边的srcID或dstID之一是活跃的，才会触发消息传递
    val sssp = initialGraph
      .pregel(initialMsg = initialMessage, maxIterations = 10, activeDirection = EdgeDirection.Either)(vprog, sendMsg, mergeMsg)

    sssp
  }

  def exploreMsgSend[V, E](graph: Graph[V, E]): Graph[Double, Int] = {

    val initialGraph: Graph[Double, Int] = graph.mapVertices((id, _) => 0.0).mapTriplets(triplet => 1)
    val initialMessage = 0.0
    val vprog = (id: VertexId, attr: Double, newMsg: Double) => (attr + newMsg)
    val sendMsg = (triplet: EdgeTriplet[Double, Int]) => {
      if(triplet.srcAttr < 2){
        if(triplet.srcId == 0){
          Iterator((triplet.dstId, 1.0)) ++ Iterator((triplet.srcId, 0.0))
        }else{
        Iterator((triplet.dstId, 1.0))
        }
      } else Iterator.empty
    }

    /*val sendMsg = (triplet: EdgeTriplet[Double, Int]) => {
      if(triplet.srcAttr < 2){

        Iterator((triplet.dstId, 1.0))

      } else Iterator.empty
    }*/
    val mergeMsg = (a: Double, b: Double) => (a + b)

    val result = initialGraph.pregel(initialMsg = initialMessage, maxIterations = 3, activeDirection = EdgeDirection.Out)(vprog, sendMsg, mergeMsg)
    result
  }

  def getBiggestNum[V, E](graph: Graph[V, E]): Graph[Double, Int] = {
    val vertexAttr = List[(VertexId, Double)]((0, 3.0), (1, 6.0), (2, 2.0), (3, 1.0)).toMap
    val newGraph = graph.mapVertices((id, _) => vertexAttr(id)).mapTriplets(triplet => 1)

    val initialMessage = Double.NegativeInfinity
    val vprog = (id: VertexId, attr: Double, newMsg: Double) => math.max(attr, newMsg)
    val sendMsg = (triplet: EdgeTriplet[Double, Int]) => {
      if(triplet.srcAttr < triplet.dstAttr) {
        Iterator((triplet.srcId, triplet.dstAttr))
      } else if(triplet.srcAttr > triplet.dstAttr) {
        Iterator((triplet.dstId, triplet.srcAttr))
      } else{
        Iterator.empty
      }
    }

    val mergeMsg = (a: Double, b: Double) => math.max(a, b)

    val result = newGraph.pregel[Double](initialMsg = initialMessage, maxIterations = Int.MaxValue, activeDirection = EdgeDirection.Either)(vprog, sendMsg, mergeMsg)
    result
  }

  def exploreMsgAgg[V, E](graph: Graph[V, E]): VertexRDD[Double] = {
    val initialGraph: Graph[Double, Int] = graph.mapVertices((id, _) => 1.0).mapTriplets(triplet => 1)
    val sendMsg = (ctx: EdgeContext[Double, Int, Map[VertexId, Double]]) => {
      ctx.sendToSrc(Map(ctx.srcId -> ctx.srcAttr))
      ctx.sendToDst(Map(ctx.dstId -> ctx.dstAttr))
      ctx.sendToDst(Map(ctx.srcId -> ctx.srcAttr))
      ctx.sendToSrc(Map(ctx.dstId -> ctx.dstAttr))
    }
    val result = initialGraph.aggregateMessages[Map[VertexId, Double]](sendMsg = sendMsg, mergeMsg = _ ++ _, tripletFields = TripletFields.All)
    result.mapValues(_.values.sum)
  }
}