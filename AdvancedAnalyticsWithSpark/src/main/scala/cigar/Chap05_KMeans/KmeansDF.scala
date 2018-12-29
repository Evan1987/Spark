package cigar.Chap05_KMeans

import cigar.Constant._
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml._
import org.apache.spark.ml.feature._
import org.apache.spark.sql._
import org.apache.spark.sql.types._
import org.apache.spark.ml.clustering.{KMeans, KMeansModel}
import scala.util.Random

object KmeansDF extends App {
  Logger.getLogger("org").setLevel(Level.ERROR)
  val sparkSession = SparkSession.builder().master("local[8]").appName("Kmeans").getOrCreate()

  val filePath = constant.PATH + "ch05-kmeans\\dataset\\kddcup.csv"
  val rawData = sparkSession
    .read
    .options(Map("inferSchema"->"true","header"->"false"))
    .csv(filePath)
    .toDF("duration", "protocol_type", "service", "flag",
      "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
      "hot", "num_failed_logins", "logged_in", "num_compromised",
      "root_shell", "su_attempted", "num_root", "num_file_creations",
      "num_shells", "num_access_files", "num_outbound_cmds",
      "is_host_login", "is_guest_login", "count", "srv_count",
      "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
      "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
      "dst_host_count", "dst_host_srv_count",
      "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
      "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
      "dst_host_serror_rate", "dst_host_srv_serror_rate",
      "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
      "label")
  //rawData.groupBy("label").count().orderBy($"count".desc).show()
  //rawData.printSchema()

  val runKmeans = new RunKmeans(sparkSession)
  runKmeans.clusteringTake0(rawData,dropString = true)
  runKmeans.clusteringTakeK(rawData)
}

class RunKmeans(private val sparkSession: SparkSession) extends Serializable {
  import sparkSession.implicits._
  // 0.1 去掉原数据中类型为String的列
  def dropStringCols(rawData:DataFrame):(DataFrame,Array[String])={
    // 去掉类型为String的列
    val stringCols = rawData.schema.filter(struct=>struct.dataType == StringType).map(x=>x.name).toArray.filter(_!="label")
    val newData = rawData.drop(stringCols:_*)
    (newData,stringCols)
  }
  // 0.2 对所需要的列做独热编码处理
  def oneHotPipeLine(inputCol:String):(Pipeline,String)={
    val indexer = new StringIndexer()
      .setInputCol(inputCol)
      .setOutputCol(inputCol + "_indexed")
    val encoder = new OneHotEncoder()
      .setInputCol(inputCol + "_indexed")
      .setOutputCol(inputCol + "_vec")
    val pipeline = new Pipeline().setStages(Array(indexer,encoder))
    (pipeline,inputCol + "_vec")
  }

  // 1.只聚合成1个cluster，流程总览
  def clusteringTake0(rawData:DataFrame,dropString:Boolean=true):Unit = {
    val data = if(dropString){dropStringCols(rawData)._1}else{rawData}

    val assembler = new VectorAssembler()
      .setInputCols(data.columns.filter(_!="label"))
      .setOutputCol("featureVector")

    val kmeans = new KMeans()
      .setSeed(Random.nextLong())
      .setPredictionCol("cluster")
      .setFeaturesCol("featureVector")

    val pipeline = new Pipeline().setStages(Array(assembler,kmeans))
    val pipelineModel = pipeline.fit(data)
    val kmeansModel = pipelineModel.stages.last.asInstanceOf[KMeansModel]
    // 输出当前质心
    kmeansModel.clusterCenters.foreach(println)
    // 对数据集进行聚类，输出每cluster内label的分布
    val withCluster = pipelineModel
      .transform(data)
      .select("cluster","label")
      .groupBy("cluster","label")
      .count()
      .orderBy($"cluster",$"count".desc)
      .show(20)
  }

  // 2.0 计算按k簇聚类后的，平均簇内距离，默认参数
  def clusteringScore0(data:DataFrame,k:Int):Double={
    val assembler = new VectorAssembler()
      .setInputCols(data.columns.filter(_!="label"))
      .setOutputCol("featureVector")

    val kmeans = new KMeans()
      .setSeed(Random.nextLong())
      .setK(k)
      .setPredictionCol("cluster")
      .setFeaturesCol("featureVector")

    val pipeline = new Pipeline().setStages(Array(assembler,kmeans))
    val pipelineModel = pipeline.fit(data)
    val kmeansModel = pipelineModel.stages.last.asInstanceOf[KMeansModel]

    // 计算（各样本点到最近cluster质心的距离平方和/数据量）的算术平方根 = 均方根（RMSE） = 平均簇内距离
    math.sqrt(kmeansModel.computeCost(assembler.transform(data))/data.count())
  }
  // 2.1 计算按k簇聚类后的，平均簇内距离，设定参数
  def clusteringScore1(data:DataFrame,k:Int):Double={
    val assembler = new VectorAssembler()
      .setInputCols(data.columns.filter(_!="label"))
      .setOutputCol("featureVector")

    val kmeans = new KMeans()
      .setSeed(Random.nextLong())
      .setK(k)
      .setPredictionCol("cluster")
      .setFeaturesCol("featureVector")
      .setMaxIter(40)//最大迭代次数
      .setTol(1.0e-5)//质心的最小变化量

    val pipeline = new Pipeline().setStages(Array(assembler,kmeans))
    val pipelineModel = pipeline.fit(data)
    val kmeansModel = pipelineModel.stages.last.asInstanceOf[KMeansModel]

    // 计算（各样本点到最近cluster质心的距离平方和/数据量）的算术平方根 = 均方根（RMSE） = 平均簇内距离
    math.sqrt(kmeansModel.computeCost(assembler.transform(data))/data.count())
  }
  // 2.2 计算按k簇聚类后的，平均簇内距离，设定参数，并对特征进行标准化处理
  def clusteringScore2(data:DataFrame,k:Int):Double={
    val assembler = new VectorAssembler()
      .setInputCols(data.columns.filter(_!="label"))
      .setOutputCol("featureVector")
    // (x-u)/sigma
    // sigma = ifelse(setWithStd,sigma,1)
    // u = ifelse(setWithMean,u,0)
    val scaler = new StandardScaler()
      .setInputCol("featureVector")
      .setOutputCol("scaledFeatureVector")
      .setWithStd(true)
      .setWithMean(true)

    val kmeans = new KMeans()
      .setSeed(Random.nextLong())
      .setPredictionCol("cluster")
      .setFeaturesCol("scaledFeatureVector")
      .setMaxIter(40)
      .setTol(1.0e-5)

    val pipeline = new Pipeline().setStages(Array(assembler,scaler,kmeans))
    val pipelineModel = pipeline.fit(data)
    val kmeansModel = pipelineModel.stages.last.asInstanceOf[KMeansModel]

    math.sqrt(kmeansModel.computeCost(pipelineModel.transform(data))/data.count())
  }
  // 2.3 计算按k簇聚类后的，平均簇内距离，设定参数，并对特征进行标准化处理，并对类别型变量进行独热编码
  def clusteringScore3(rawData:DataFrame,stringCols:Array[String],k:Int):Double={
    // 对所有类别型变量生成独热处理PipeLine列表 和 输出列名列表
    val oneHotList = stringCols.map(col=>oneHotPipeLine(col))
    val oneHotPipelines = oneHotList.map(_._1)
    val oneHotCols = oneHotList.map(_._2)

    val newCols = rawData.columns.toSet.--(stringCols).++(oneHotCols).toArray

    val assembler = new VectorAssembler()
      .setInputCols(newCols.filter(_!="label"))
      .setOutputCol("featureVector")
    // (x-u)/sigma
    // sigma = ifelse(setWithStd,sigma,1)
    // u = ifelse(setWithMean,u,0)
    val scaler = new StandardScaler()
      .setInputCol("featureVector")
      .setOutputCol("scaledFeatureVector")
      .setWithStd(true)
      .setWithMean(true)
    val kmeans = new KMeans()
      .setSeed(Random.nextLong())
      .setPredictionCol("cluster")
      .setFeaturesCol("scaledFeatureVector")
      .setMaxIter(40)
      .setTol(1.0e-5)


    val pipeline = new Pipeline().setStages(oneHotPipelines.++(Array(assembler,scaler,kmeans)))
    val pipelineModel = pipeline.fit(rawData)

    val kmeansModel = pipelineModel.stages.last.asInstanceOf[KMeansModel]
    math.sqrt(kmeansModel.computeCost(pipelineModel.transform(rawData))/rawData.count())
  }

  // 2.批量计算不同k值下，不同条件下平均簇内距离
  def clusteringTakeK(rawData:DataFrame):Unit={
    val (data,stringCols) = dropStringCols(rawData)
    data.cache()
    // 并行化处理，par
    println("scores in default pars")
    //(20 to 100 by 20).par.map(k=>(k,clusteringScore0(data,k))).foreach(println)
    (20 to 100 by 20).map(k=>(k,clusteringScore0(data,k))).foreach(println)

    println("scores in some pars")
    (20 to 100 by 20).map(k=>(k,clusteringScore1(data,k))).foreach(println)

    println("scores in some pars and standarized features")
    (20 to 100 by 20).map(k=>(k,clusteringScore2(data,k))).foreach(println)

    println("scores in some pars, standarized features and one-hoted category cols")
    (20 to 100 by 20).map(k=>(k,clusteringScore3(rawData,stringCols,k))).foreach(println)
    data.unpersist()
  }
}
