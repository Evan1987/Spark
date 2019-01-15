
package cigar.Chap07_GraphX

import cigar.Constant._

import scala.xml._
import java.io.{File, PrintWriter}

import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.SparkSession
import org.apache.spark.graphx._
import com.google.common.hash.Hashing
import org.apache.spark.rdd.RDD

import scala.reflect.ClassTag


object GraphX {

  val filePath = constant.PATH + "ch07-graph\\data\\"
  val meshTablePath = filePath + "mesh.txt"
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)

    // ******************* 0.处理数据集 *******************
    if(!new File(meshTablePath).exists()){
      println("Generate MeshTable!")
      generateMeshTable(filePath, meshTablePath)
    }else{
      println("Target MeshTable already exists!")
    }

    val spark = SparkSession.builder().master("local[4]").appName("GraphX").getOrCreate()

    // 去掉没有Mesh属性的数据，并将字符串进行拆分
    // row为Mesh属性列表
    val topicLog = spark.sparkContext.textFile(meshTablePath).flatMap(row => {
      val arr = row.split("\t")
      if(arr.length != 2){
        List()
      }else{
        List(arr(1).split(",").map(_.stripPrefix(" ").stripSuffix(" ")).toList)
      }
    })

    // ******************* 1. 建立图 *******************
    // 建立顶点信息， (Vertex_ID， VD)
    val vertices = topicLog.flatMap(x => x).map(x => (hashID(x), x))  // 这里可以不去重，在构建图时会自动去重
    //println(vertices.count())  // 300680

    // 建立边信息，这里主要是共现信息 (pair, cooccur_num)
    val cooccurs: RDD[(List[String], Int)] = topicLog.flatMap(topics => topics.sorted.combinations(2)).map(pair => (pair, 1)).reduceByKey(_ + _)
    // cooccurs.top(10)(Ordering.by[(List[String], Int), Int](_._2)).foreach(println)  // RDD里的top取的是ord.reverse (坑！！)

    val edges = cooccurs.map{case (pair, cnt) => {
      val ids = pair.map(hashID).sorted
      Edge(ids(0), ids(1), cnt)  // Edge信息，pair已经sort过了
    }}
    //println(edges.count())  //233829

    // 建立图，由顶点信息和边组成
    val topicGraph: Graph[String, Int] = Graph(vertices, edges)  // 顶点属性是 string
    topicGraph.cache()
    //println(topicGraph.vertices.count())  // 13522
    // vertices: (VertexID: Long & attribute: String)
    // edges: (srcID: Long, dstID: Long & attribute: Int)
    // triplets: srcVertex, dstVertex, edge

    // ******************* 2. 连通图 *******************
    // 2.1 连通图探查（任意两点间有路径相连）
    // 获取连通组件所含顶点的数量
    val connectedComponentGraph: Graph[VertexId, Int] = topicGraph.connectedComponents()  // 顶点属性是一个 VertexId  (本体VertexID, 所在连通图的最小VertexID)
    val componentCounts: Seq[(VertexId, Long)] = connectedComponentGraph.vertices.map(x => x._2).countByValue().toSeq.sortBy(_._2).reverse
    //println(componentCounts.length)  // 连通组件数  647
    //componentCounts.take(10).foreach(println)  // 包含定点数最多的前10个组件及其顶点数
    /*
    (-9222594773437155629,12820)
    (-6100368176168802285,5)
    (-1043572360995334911,4)
    (-8248220249038703481,3)
    (-8082131391550700575,3)
    (-7698783380355679916,3)
    (-6561074051356379043,3)
    (-8186497770675508345,3)
    (5624290286186907719,2)
    (-7965984711083374415,2)*/


    // left VertexRDD[String] innerJoin right VertexRDD[VertexID] on left vertexID == right vertexID(topicID)
    // return vertexID a new vertex attribute
    // function (right vertexID, left_attribute, right_attribute) => new attribute (String, ComponentID)
    val nameCID: VertexRDD[(String, VertexId)] = (topicGraph.vertices).innerJoin(connectedComponentGraph.vertices)((topicID, name, componentID) => (name, componentID))

    // 2.2 探查排名第二位的连通组件主题名称
    val c1 = nameCID.filter{case (vertexID, (name, componentID)) => (componentID == componentCounts(1)._1)}
    //c1.collect().foreach{case (vertexID, (name, componentID)) => println(name)}
    /*
    Acetyl-CoA C-Acyltransferase
    Enoyl-CoA Hydratase
    3-Hydroxyacyl CoA Dehydrogenases
    Racemases and Epimerases
    Carbon-Carbon Double Bond Isomerases*/

    // 2.3 连通图顶点度的分布
    // 每个顶点的度
    val degrees: VertexRDD[Int] = topicGraph.degrees
    //println(degrees.map{case (vertexID, degree) => degree}.stats())  //(count: 12922, mean: 36.190837, stdev: 76.318279, max: 2634.000000, min: 1.000000)

    val nameAndDegrees: VertexRDD[(String, Int)] = degrees.innerJoin(topicGraph.vertices)((topicID, degree, name) => (name, degree))
    //nameAndDegrees.top(10)(Ordering.by[(VertexId, (String, Int)), Int](_._2._2)).foreach(println)
    /*(3610584643062193094,(Research,2634))
    (-3877629166510072267,(Disease,1788))
    (2424418520274103030,(Neoplasms,1397))
    (-3605489290966697873,(Diagnosis,975))
    (-15642666922855628,(Blood,943))
    (-8526825853681895865,(Medical,929))
    (-1752716418447087129,(Models,918))
    (-363892805528295733,(Pharmacology,903))
    (5298123727603740895,(Tuberculosis,868))
    (1039115496958927634,(Infant,812))*/


    //  ******************* 3. 过滤噪声边 *******************
    // 测试一个属性的出现与另一个属性的出现是否独立（e.g. 成对出现是碰巧还是显著的） -> 卡方独立性检验
    val total = topicLog.count()  // 总频数
    val topicCounts = vertices.map{case (vertexID, name) => (vertexID, 1)}.reduceByKey(_ + _)  // 各属性出现的次数
    val topicCountsGraph: Graph[Int, Int] = Graph(topicCounts, topicGraph.edges)  // 生成顶点属性为出现频次的新图

    // 生成顶点属性仍是频次， 边属性为卡方值的新图
    val chiSquareGraph: Graph[Int, Double] = topicCountsGraph.mapTriplets(triplet => {
      chiSquare(total, triplet.dstAttr, triplet.srcAttr, triplet.attr)
    })

    //println(chiSquareGraph.edges.map(x => x.attr).stats())  // (count: 233829, mean: 456.968916, stdev: 3228.356097, max: 106296.559840, min: 0.000000)

    // 取p值大于0.99999（卡方 > 19.5）的子图（过滤掉p值低于0.99999的边）
    val interesting = chiSquareGraph.subgraph(triplet => triplet.attr > 19.5)
    //println(interesting.edges.count())  //111398

    // 探查子图
    // 子图的连通性
    val interestingComponents = interesting.connectedComponents()
    val interestingComponentsCounts = interestingComponents.vertices.map{case (vertexID, attrID) => (attrID, 1)}.countByValue().toSeq.sortBy(_._2).reverse
    //println(interestingComponentsCounts.length)  // 657  相比原来增加了10个连通件，关系影响不大
    //interestingComponentCounts.take(10).foreach(println)

    // 子图的顶点度
    val interestingDegrees = interesting.degrees
    //println(interestingDegrees.map{case (vertexID, degree) => degree}.stats())  // (count: 12912, mean: 17.254957, stdev: 22.007308, max: 494.000000, min: 1.000000)


    //  ******************* 4. "小世界"网络指标 *******************
    // "顶点V的三角计数" t => 顶点V有多少邻接点是相互连接的
    // "局部聚类系数" C => C = 2t / k(k - 1)   by Watts & Strogatz, k: 顶点V的邻接点数
    val triCountGraph: Graph[Int, Double] = interesting.triangleCount()  // 顶点属性变为三角计数
    //println(triCountGraph.vertices.map{case (vertexID, triCount) => triCount}.stats())  // (count: 13522, mean: 50.503698, stdev: 177.286792, max: 4326.000000, min: 0.000000)

    val maxTrisGraph = interesting.degrees.mapValues(degree => degree * (degree - 1) / 2.0)
    val clusterCoefGraph: VertexRDD[Double] = triCountGraph.vertices.innerJoin(maxTrisGraph){(vertexID, triCount, maxTris) => if(maxTris == 0) 0 else triCount / maxTris}  // 顶点属性为局部聚类系数
    //println(clusterCoefGraph.map{case (vertexID, coef) => coef}.sum() / clusterCoefGraph.count())  // 平均聚类系数 0.31213166074395043


    // 计算平均路径长度
    val paths = samplePathLengths(interesting, seed = 1792L, sampleRatio = 0.02)
    //println(paths.map{case (srcID, dstID, length) => length}.filter(_ > 0).stats())  // (count: 3447193, mean: 3.792891, stdev: 0.786798, max: 9.000000, min: 1.000000)

    val hist = paths.map{case (srcID, dstID, length) => length}.countByValue()
    /*hist.toSeq.sortBy(_._1).foreach(println)
    (0,283)
    (1,5296)
    (2,122862)
    (3,1043655)
    (4,1734007)
    (5,495218)
    (6,42498)
    (7,3521)
    (8,135)
    (9,1)*/
  }


  /**
    * 生成解析文件，并保存在 meshTablePath文件夹下
    * */
  def generateMeshTable(sourcePath: String, meshTablePath: String): Unit = {

    val filePath = new File(sourcePath)
    val xmlFiles = filePath.listFiles()
    val fileNum = xmlFiles.length

    val writer = new PrintWriter(meshTablePath)

    xmlFiles.zipWithIndex.foreach {case(file, index) => {
      val xmlFile = XML.loadFile(file)
      val docInfos = ParseXML.getDocTopicFromFile(xmlFile)
      for (docInfo <- docInfos) {
        val id = docInfo.id
        val topics = docInfo.topics
        writer.println(id + "\t" + topics.mkString(","))
      }
      println(s"[${index + 1}/$fileNum]  Save ${file.getName} completed, row num: ${docInfos.length.toString}")
    }}
    writer.close()
    println("-------All work done!---------")
  }

  def hashID(s: String): Long = {
    Hashing.md5().hashUnencodedChars(s).asLong()
  }

  /**
    * 计算边的卡方值，采用修正公式 T * ((ad - bc)**2 - T / 2) / ((a + b)(a + c)(b + d)(c + d))
    * param:
    * - total: 数据总频数
    * - ya: 端点a的频数
    * - yb: 端点b的频数
    * - yy: 端点a和b共现的频数
    * */
  def chiSquare(total: Long, ya: Int, yb: Int, yy: Int): Double = {
    val na = total - ya
    val nb = total - yb
    val yn = ya - yy
    val ny = yb - yy
    val nn = total - yy - yn - ny
    total * math.pow((yy * nn - yn * ny) - total / 2.0, 2.0) / (ya * yb * na * nb)
  }

  /**
    * Pregel 计算平均路径的主函数
    * */
  def samplePathLengths[V: ClassTag, E: ClassTag](graph: Graph[V, E], seed: Long, sampleRatio: Double): RDD[(VertexId, VertexId, Int)] = {
    val sample = graph.vertices.map{case (vertexID, attr) => vertexID}.sample(withReplacement = false, fraction=sampleRatio, seed = seed)
    val ids: Set[VertexId] = sample.collect().toSet
    val mapGraph = graph.mapVertices((id, v) => if(ids.contains(id)) Map(id -> 0) else Map[VertexId, Int]())

    val initial = Map[VertexId, Int]()  // 初始化消息，初始化图中各节点的属性
    val res = mapGraph.ops.pregel(initialMsg = initial)(vprog = update, sendMsg = iterate, mergeMsg = mergeMaps)
    res.vertices.flatMap{case (id, m) =>
      m.map{case (k, v) =>
        if (id < k) (id, k, v) else (k, id, v)
      }
    }.distinct()
  }


  /**
    * 消息合并函数
    * */
  def mergeMaps(m1: Map[VertexId, Int], m2: Map[VertexId, Int]): Map[VertexId, Int] = {
    def minThatExists(k: VertexId): Int = {
      math.min(m1.getOrElse(k, Int.MaxValue), m2.getOrElse(k, Int.MaxValue))
    }

    (m1.keySet ++ m2.keySet).map(k => (k, minThatExists(k))).toMap
  }

  /**
    * 节点变换函数
    * 对收到消息的节点进行消息的更新变换
    * */
  def update(id: VertexId, state: Map[VertexId, Int], msg: Map[VertexId, Int]): Map[VertexId, Int] = {
    mergeMaps(state, msg)
  }

  def checkIncrement(a: Map[VertexId, Int], b: Map[VertexId, Int], bid: VertexId): Iterator[(VertexId, Map[VertexId, Int])] = {
    val aplus = a.map{case (v, attr) => (v -> (attr + 1))}
    if (b != mergeMaps(aplus, b)) {
      Iterator((bid, aplus))
    } else {
      Iterator.empty
    }
  }

  /**
    * 消息发送函数，决定给哪些节点发送哪些消息
    *
    * */
  def iterate(e: EdgeTriplet[Map[VertexId, Int], _]): Iterator[(VertexId, Map[VertexId, Int])] = {
    checkIncrement(e.srcAttr, e.dstAttr, e.dstId) ++ checkIncrement(e.dstAttr, e.srcAttr, e.srcId)

  }

}


