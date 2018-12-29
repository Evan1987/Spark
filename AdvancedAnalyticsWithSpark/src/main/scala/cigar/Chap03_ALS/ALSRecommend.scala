package cigar.Chap03_ALS

import cigar.Constant._
import org.apache.log4j.{Level, Logger}
import scala.collection.Map
import scala.collection.mutable.ArrayBuffer
import scala.util.Random
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.recommendation.{ALS, ALSModel}
import org.apache.spark.sql._
import org.apache.spark.sql.functions._


object ALSRecommend extends App {
  Logger.getLogger("org").setLevel(Level.ERROR)
  val sparkSession = SparkSession.builder().master("local[8]").appName("ALSRecommend").getOrCreate()
  val path = constant.PATH + "ch03-recommender\\profiledata_06-May-2005\\"

  val rawUserArtistData = sparkSession.read.textFile(path + "sample_user_artist_data.txt")
  val rawArtistData = sparkSession.read.textFile(path + "artist_data.txt")
  val rawArtistAliasData = sparkSession.read.textFile(path + "artist_alias.txt")


  val recommender = new ALSRecommender(sparkSession,rawUserArtistData,rawArtistData,rawArtistAliasData)
  // 1.训练结果
  val model = recommender.modeling()
  model.userFactors.show(10)
  model.itemFactors.show(10)
  // 2.对某一用户生成Top10推荐结果
  val recommendResult = recommender.makeRecommendations(model, 1048941, 10)
      .select("artist")
      .withColumnRenamed("artist","id")
      .join(recommender.artistData,"id")
  recommendResult.show()
  // 3.做模型评估
  recommender.evaluate()
}


class ALSRecommender(sparkSession: SparkSession,
                     rawUserArtistData:Dataset[String],
                     rawArtistData:Dataset[String],
                     rawArtistAliasData:Dataset[String]) extends Serializable {
  import sparkSession.implicits._
  /*0.数据清洗过程*/
  /*0.1 对 artist_data 做数据清洗与转换*/
  def buildArtistData(rawArtistData: Dataset[String]):DataFrame={
    rawArtistData.flatMap(line=>{
      val (id,name) = line.span(_!='\t')
      if(name.isEmpty){
        None
      }else{
        try{
          Some((id.toInt,name.trim))
        }catch {
          case _:NumberFormatException => None
        }
      }
    }).toDF("id", "name")
  }

  /*0.2 对 artist_alias 做数据清洗与转换，collect成Map形式*/
  def buildArtistAliasData(rawArtistAliasData: Dataset[String]):Map[Int,Int]={
    rawArtistAliasData.flatMap(line=>{
      val Array(artist, alias) = line.split('\t')
      if(artist.isEmpty){
        None
      }else{
        try{
          Some((artist.toInt, alias.toInt))
        }catch{
          case _:NumberFormatException => None
        }
      }
    }).collect().toMap
  }

  /*0.3 对 user_artist_alias 做数据清洗->利用广播变量bArtistAlias对artistID进行清洗*/
  def buildUserArtistData(rawUserArtistData:Dataset[String], bArtistAlias:Broadcast[Map[Int,Int]]):DataFrame={
    rawUserArtistData.flatMap(line=>{
      val Array(userID, artistID, count) = line.split(' ')
      if(userID.isEmpty || artistID.isEmpty || count.isEmpty){
        None
      }else{
        try{
          val finalArtistID = bArtistAlias.value.getOrElse(artistID.toInt, artistID.toInt)
          Some((userID.toInt, finalArtistID, count.toInt))
        }catch{
          case _:NumberFormatException => None
        }
      }
    }).toDF("user", "artist", "count")
  }
  /*0.4 数据清洗结果*/
  val artistData = buildArtistData(rawArtistData)
  val artistAliasData = buildArtistAliasData(rawArtistAliasData)
  val bArtistAlias = sparkSession.sparkContext.broadcast(artistAliasData)
  val userArtistData = buildUserArtistData(rawUserArtistData,bArtistAlias).cache()

  val allArtistIDs = userArtistData.select("artist").as[Int].distinct().collect()
  val bAllArtistIDs = sparkSession.sparkContext.broadcast(allArtistIDs)


  /*1. 建模*/
  def modeling():ALSModel={
    val model = new ALS()
      .setSeed(Random.nextLong())
      .setImplicitPrefs(true)
      .setRank(5)
      .setRegParam(0.01)
      .setAlpha(1.0)
      .setMaxIter(5)
      .setUserCol("user")
      .setItemCol("artist")
      .setRatingCol("count")
      .setPredictionCol("prediction")
      .fit(userArtistData)

    userArtistData.unpersist()
    model
  }
  /*2. 生成TopN推荐（可能包含已经听过的）*/
  def makeRecommendations(model:ALSModel, userID:Int, num:Int):DataFrame={
    // 获得全体 ItemID->ArtistID
    val toRecommend = model
      .itemFactors
      .select($"id".as("artist"))
      .withColumn("user", lit(userID))
    // 将transform的dataframe left join模型内的
    // userFactor和itemFactor来计算推荐分数（prediction）
    model.transform(toRecommend)
      .select("artist","prediction")
      .orderBy($"prediction".desc)
      .limit(num)
  }


  /*3. 模型评估*/
  //3.1 AUC计算：用的是AUC的概率意义而不是传统的根据ROC曲线，步进去累加计算。
  def auc(positiveData:DataFrame,
          bAllArtistIDs:Broadcast[Array[Int]],
          predictFunction:(DataFrame=>DataFrame)):Double={
    // 1. 对正样本的预测
    val positivePredictions = predictFunction(positiveData.select("user","artist"))
      .withColumnRenamed("prediction","positivePrediction")
    // 2. 对负样本的预测
    // 2.1 根据真实数据生成用户实际未听过的艺术家，负样本。
    val negativeData = positiveData.select("user","artist")
      .as[(Int,Int)]
      .groupByKey{case (user,_)=>user}
      .flatMapGroups{
        case (userID, userIDAndPosArtistIDs)=>
          {
            val random = new Random()
            // 同一用户的听过的艺术家
            val posItemIDSet = userIDAndPosArtistIDs.map{case (_,artist)=>artist}.toSet
            val negative = ArrayBuffer[Int]()
            val allArtistIDs = bAllArtistIDs.value
            var i = 0
            while (i<allArtistIDs.length && negative.size<posItemIDSet.size){
              // 随机选择一个artistID
              val artistID = allArtistIDs(random.nextInt(allArtistIDs.length))
              if(!posItemIDSet.contains(artistID)){
                negative.append(artistID)
              }
              i += 1
            }
            negative.map(artistID => (userID, artistID))
          }
      }.toDF("user","artist")
    //2.2 对负样本进行预测
    val negativePredictions = predictFunction(negativeData)
      .withColumnRenamed("prediction","negativePrediction")
    //3. 得到正负联合的全数据
    val joinedPredictions = positivePredictions
      .join(negativePredictions,"user")
      .select("user","positivePrediction","negativePrediction").cache()

    //4. 计算AUC
    val allCounts = joinedPredictions.groupBy("user")
      .agg(count(lit("1")).as("total"))
      .select("user","total")
    val correctCounts = joinedPredictions.filter($"positivePrediction">$"negativePrediction")
      .groupBy("user")
      .agg(count(lit("1")).as("correct"))
      .select("user","correct")

    // auc的意义就是正向样本排在负向样本前面的概率
    val meanAUC = allCounts.join(correctCounts,"user")
      .select($"user",($"correct"/$"total").as("auc"))
      .agg(mean($"auc"))
      .as[Double]
      .first()

    joinedPredictions.unpersist()
    meanAUC
  }

  //3.2 利用AUC指标对不同超参数进行评估，同时采用一种比较不错的简单方法作为基准
  def evaluate():Unit={
    val Array(train,test) = userArtistData.randomSplit(Array(0.9,0.1))
    train.cache()
    test.cache()
    val mostListenedAUC = auc(test, bAllArtistIDs, predictMostListened(train))
    println("the Most Listen Method's AUC is " + mostListenedAUC)

    val evaluations =
      for(rank <- Seq(5,10);
          regParam <- Seq(1.0,0.0001);
          alpha <- Seq(1.0,40.0))
      yield{
        val model = new ALS()
          .setSeed(Random.nextLong())
          .setImplicitPrefs(true)
          .setRank(rank)
          .setRegParam(regParam)
          .setAlpha(alpha)
          .setMaxIter(20)
          .setUserCol("user")
          .setItemCol("artist")
          .setRatingCol("count")
          .setPredictionCol("prediction")
          .fit(train)
        val aucValue = auc(test, bAllArtistIDs, model.transform)

        model.userFactors.unpersist()
        model.itemFactors.unpersist()

        (aucValue, (rank, regParam, alpha))
      }
    evaluations.sortBy(_._1).reverse.foreach(println)
    train.unpersist()
    test.unpersist()
  }



  /*基准比对，将听过最多的歌曲推给用户*/
  def predictMostListened(train:DataFrame)(allData:DataFrame):DataFrame={
    val listenCounts = train.groupBy("artist")
      .agg(sum("count").as("prediction"))
      .select("artist","prediction")
    allData.join(listenCounts,Seq("artist"),"left")
      .select("user","artist","prediction")
  }


}