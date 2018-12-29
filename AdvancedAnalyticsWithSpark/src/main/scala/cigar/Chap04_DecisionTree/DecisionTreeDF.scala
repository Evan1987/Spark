
package cigar.Chap04_DecisionTree

import cigar.Constant._
import org.apache.spark.ml._
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.classification.{DecisionTreeClassifier, RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorAssembler, VectorIndexer}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.sql._
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.ml.util.{Identifiable}
import scala.util.Random
import org.apache.log4j.{Level, Logger}


object DecisionTreeDF extends App {
  Logger.getLogger("org").setLevel(Level.ERROR)
  val sparkSession = SparkSession.builder().master("local[4]").appName("DecisionTree").getOrCreate()
  import sparkSession.implicits._
  val filePath = constant.PATH + "ch04-rdf\\dataset\\covtype.data"
  val dataWithOutHeader = sparkSession.read.format("com.databricks.spark.csv")
    .option("inferSchema","true")
    .option("header","false")
    .load(filePath)
  /*为创建的DataFrame设置列名*/
  val colNames =
    Seq("Elevation", "Aspect", "Slope",
      "Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology",
      "Horizontal_Distance_To_Roadways",
      "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm",
      "Horizontal_Distance_To_Fire_Points") ++
      ((0 until 4).map(i=>s"Wilderness_Area_$i")) ++
      ((0 until 40).map(i=>s"Soil_Type_$i")) ++
      Seq("Cover_Type")

  val data = dataWithOutHeader.toDF(colNames = colNames:_*).withColumn("Cover_Type",$"Cover_Type".cast("Double"))
/*  println("the data's schema is:")
  data.printSchema()
  data.show(10)*/
  //划分训练集和测试集
  val Array(trainData, testData) = data.randomSplit(Array(0.9,0.1))
  trainData.cache(); testData.cache()

  /*决策树算法*/
  val runDecisionTree = new DecisionTreeDF(sparkSession)
  //1.简单决策树展示
  runDecisionTree.simpleDecisionTree(trainData, testData)
  val randomAccuracy = runDecisionTree.randomClassifier(trainData,testData)
  println(s"the random guess accuracy is: $randomAccuracy")
  //2.最优参数搜索的决策树展示
  runDecisionTree.bestDTModeling(trainData,testData,unOneHot = true)
  //3.最优参数搜索的随机森林展示
  //runDecisionTree.randomForrest(trainData,testData,unOneHot = true)

  trainData.unpersist(); testData.unpersist()
}

class DecisionTreeDF(private val sparkSession:SparkSession){
  import sparkSession.implicits._
  /*0.取消原数据的独热编码，减小内存使用*/
  def unencodeOneHot(data:DataFrame):DataFrame={
    // 各onehot编码的列名
    val wildernessCols = (0 until 4).map(i=>s"Wilderness_Area_$i").toArray
    val soilCols = (0 until 40).map(i=>s"Soil_Type_$i").toArray
    // 转换UDF，目标列已经转换为[0,1,0,0]Vector格式，每个Vector内只含1个1.0，用1.0的索引来代替此列
    val unhotUDF = udf((vec:Vector)=>vec.toArray.indexOf(1.0).toDouble)

    val wildernessAssembler = new VectorAssembler().setInputCols(wildernessCols).setOutputCol("wilderness")
    // 生成一个新的dataframe，在data的基础上多了一个组合列（wilderness），类型为Vector
    val withWilderness_pre = wildernessAssembler.transform(data)
    // 选取最终要留下的列名（不在oneHot编码列名里）
    val wildernessSelectCols = withWilderness_pre.columns.toSet.diff(wildernessCols.toSet).toArray.map(col(_))
    val withWilderness = withWilderness_pre.select(wildernessSelectCols:_*).withColumn("wilderness",unhotUDF($"wilderness"))

    val soilAssembler = new VectorAssembler().setInputCols(soilCols).setOutputCol("soil")
    val withSoil_pre = soilAssembler.transform(withWilderness)
    val soilSelectCols = withSoil_pre.columns.toSet.diff(soilCols.toSet).toArray.map(col(_))
    val withSoil = withSoil_pre.select(soilSelectCols:_*).withColumn("soil",unhotUDF($"soil"))
    withSoil
  }
  /*1.简单决策树实现*/
  // 默认参数、默认数据集，工作流、测试、性能评价等流程展示
  def simpleDecisionTree(trainData:DataFrame, testData:DataFrame, unOneHot:Boolean=false) : Unit = {
    /*0.选择是否取消独热*/
    val Array(train,test) =
      if(unOneHot){
        Array(trainData,testData).map(unencodeOneHot(_))
      }else{
        Array(trainData,testData)
      }
    /*1.构建数据流*/
    //stage1: 类别转换为索引
    val labelIndexer = new StringIndexer()
      .setInputCol("Cover_Type")
      .setOutputCol("labeled_Cover_Type")
      .fit(train)
    //stage2：设定特征列
    val assembler = new VectorAssembler()
      .setInputCols(train.columns.filter(_!="Cover_Type"))
      .setOutputCol("featureVector")
    //stage3：建立决策树模型
    val classifier = new DecisionTreeClassifier()
      .setSeed(Random.nextLong())
      .setLabelCol("labeled_Cover_Type")
      .setFeaturesCol("featureVector")
      .setPredictionCol("labeled_prediction")
      .setMaxBins(10)
    //stage4：反转索引回类别
    val labelConverter = new IndexToString()
      .setInputCol("labeled_prediction")
      .setOutputCol("predictionStr")
      .setLabels(labelIndexer.labels)
    //stage5：转化输出类别的类型 string->Double
    val formatConverter = new ColFormatConverter()
      .setInputCol("predictionStr")
      .setOutputCol("prediction")

    // 根据以上各stage建立工作流
    val pipeline = new Pipeline().setStages(Array(labelIndexer,assembler,classifier,labelConverter,formatConverter))
    val model = pipeline.fit(train)
    // 输出预测结果，结果中除了pipeline中新增的outputcol外，由算法模型直接输出的还有”probability“
    val predictions = model.transform(train)
    predictions.cache()
    // 特征重要度打印，貌似只有spark 2以上版本才能将DecisionTreeModel转成RandomForestModel
    //model.stages(2).asInstanceOf[RandomForestClassificationModel].featureImportances.toArray.zip(trainData.columns).sortBy(_._1).reverse.foreach(println)

    /*2.输出分类的性能评价*/
    /*2.1、precision，recall，f1值*/
    /*对于多分类问题，单个指标无效，整体只能看看准确率。精度和召回必须精确到每个类别*/
    val evaluator = new MulticlassClassificationEvaluator().setPredictionCol("prediction").setLabelCol("Cover_Type")
    val precision = evaluator.setMetricName("weightedPrecision").evaluate(predictions)
    val recall = evaluator.setMetricName("weightedRecall").evaluate(predictions)
    val f1 = evaluator.setMetricName("f1").evaluate(predictions)
    println(s"the predict precision is $precision\t,the recall is $recall\t,the f1 value is $f1")
    /*2.2、混淆矩阵*/
    /*Method：1 利用MulticlassMetrics方法输出混淆矩阵，MulticlassMetrics只能接受RDD[LabeledPoint]*/
    val predictionsRDD = predictions.select("prediction","Cover_Type").as[(Double,Double)].rdd
    val multiclassMetrics = new MulticlassMetrics(predictionsRDD)
    println(s"confusion matrix is:")
    println(multiclassMetrics.confusionMatrix)
    /*Method：2 利用透视表模拟输出混淆矩阵，指定“prediction”为透视列，且可能取值为Seq：（1 to 7）->为了提高计算性能*/
    val confusionMatrix = predictions.groupBy("Cover_Type")
      .pivot("prediction",(1 to 7))
      .count().na.fill(0.0).orderBy("Cover_Type")
    println(s"another method to output matrix is:")
    confusionMatrix.show()

    /*3.测试集效果*/
    val testPredictions = model.transform(test)
    val testAccuracy = evaluator.setMetricName("weightedPrecision").evaluate(testPredictions)
    println(s"the simple DT's accuracy on testDF is: $testAccuracy")
  }
  /*2.在参数网格中挑选最佳参数获得最佳DT模型*/
  def bestDTModeling(trainData:DataFrame,testData:DataFrame,unOneHot:Boolean=false):Unit={
    /*0.选择是否取消独热*/
    val Array(train,test) =
      if(unOneHot){
        Array(trainData,testData).map(unencodeOneHot(_))
      }else{
        Array(trainData,testData)
      }

    /*1.构建数据流*/
    //stage1: 类别转换为索引
    val labelIndexer = new StringIndexer()
      .setInputCol("Cover_Type")
      .setOutputCol("labeled_Cover_Type")
      .fit(train)
    //stage2：设定特征列
    val assembler = new VectorAssembler()
      .setInputCols(train.columns.filter(_!="Cover_Type"))
      .setOutputCol("featureVector")
    //stage3：建立决策树模型
    val classifier = new DecisionTreeClassifier()
      .setSeed(Random.nextLong())
      .setLabelCol("labeled_Cover_Type")
      .setFeaturesCol("featureVector")
      .setPredictionCol("labeled_prediction")
      .setMinInstancesPerNode(5)
    //stage4：反转索引回类别
    val labelConverter = new IndexToString()
      .setInputCol("labeled_prediction")
      .setOutputCol("predictionStr")
      .setLabels(labelIndexer.labels)
    //stage5：转化输出类别的类型 string->Double
    val formatConverter = new ColFormatConverter()
      .setInputCol("predictionStr")
      .setOutputCol("prediction")
    val pipeLine = new Pipeline().setStages(Array(labelIndexer,assembler,classifier,labelConverter,formatConverter))

    /*2.构建最佳训练模型*/
    // 2.1 参数网格
    val paramGrid = new ParamGridBuilder()
      .addGrid(classifier.impurity,Seq("entropy","gini"))
      .addGrid(classifier.maxDepth,Seq(1,20))
      .addGrid(classifier.maxBins,Seq(40,300))
      .addGrid(classifier.minInfoGain,Seq(0.0,0.05))
      .build()

    // 2.2 模型评估器
    val evaluator =
      new MulticlassClassificationEvaluator()
        .setLabelCol("Cover_Type")
        .setPredictionCol("prediction")
        .setMetricName("weightedPrecision")

    // 2.3 模型评估总流程，容纳工作流、评估器、参数网格及其他训练参数
    //建立基于参数网格的单CV测试，与CrossValidator不同
    val validator = new TrainValidationSplit()
      .setEstimator(pipeLine)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.9)

    // 2.4 多参数的模型训练，获取最佳模型和参数
    val validatorModel = validator.fit(train)
    // 根据evaluator定义的评价方式，输出所有参数模型的评估结果
    val accuracy = validatorModel.validationMetrics
    // 根据最好的评估结果得到最好的模型和相应的参数
    val bestModel = validatorModel.bestModel
    val params:Array[ParamMap] = validatorModel.getEstimatorParamMaps
    val bestParams = params(accuracy.indexOf(accuracy.max))
    println("the best Params is:",bestParams)

    /*3.模型预测*/
    // 在最终测试集上进行预测，得到测试准确率
    val predictions = bestModel.transform(test)
    val testAccuracy = evaluator.evaluate(predictions)
    println(s"the best DT model's accuracy on testDF is: $testAccuracy")
  }
  /*3.在参数网格中挑选最佳参数获得最佳随机森林模型*/
  def randomForrest(trainData:DataFrame,testData:DataFrame,unOneHot:Boolean=false):Unit = {

    /*0.选择是否取消独热*/
    val Array(train,test) =
      if(unOneHot){
        Array(trainData,testData).map(unencodeOneHot(_))
      }else{
        Array(trainData,testData)
      }

    /*1.构建数据流*/
    //stage1: 类别转换为索引
    val labelIndexer = new StringIndexer()
      .setInputCol("Cover_Type")
      .setOutputCol("labeled_Cover_Type")
      .fit(train)
    //stage2：设定特征列
    val assembler = new VectorAssembler()
      .setInputCols(train.columns.filter(_!="Cover_Type"))
      .setOutputCol("featureVector")
    //**stage3：建立随机森林模型
    val classifier = new RandomForestClassifier()
      .setSeed(Random.nextLong())
      .setLabelCol("labeled_Cover_Type")
      .setFeaturesCol("featureVector")
      .setPredictionCol("labeled_prediction")
      .setImpurity("gini")
      .setMaxDepth(20)
      .setMaxBins(300)
      .setMinInstancesPerNode(5)
    //stage4：反转索引回类别
    val labelConverter = new IndexToString()
      .setInputCol("labeled_prediction")
      .setOutputCol("predictionStr")
      .setLabels(labelIndexer.labels)
    //stage5：转化输出类别的类型 string->Double
    val formatConverter = new ColFormatConverter()
      .setInputCol("predictionStr")
      .setOutputCol("prediction")

    val pipeLine = new Pipeline().setStages(Array(labelIndexer,assembler,classifier,labelConverter,formatConverter))

    /*2.构建最佳训练模型*/
    // 2.1 参数网格
    val paramGrid = new ParamGridBuilder()
      .addGrid(classifier.minInfoGain, Seq(0.0, 0.05))
      .addGrid(classifier.numTrees, Seq(1, 10))
      .build()

    // 2.2 模型评估器
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("Cover_Type")
      .setPredictionCol("prediction")
      .setMetricName("weightedPrecision")

    // 2.3 模型评估总流程，容纳工作流、评估器、参数网格及其他训练参数
    val validator = new TrainValidationSplit()
      .setEstimator(pipeLine)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.9)

    // 2.4 多参数的模型训练，获取最佳模型和参数
    val validatorModel = validator.fit(train)
    // 根据evaluator定义的评价方式，输出所有参数模型的评估结果
    val accuracy:Array[Double] = validatorModel.validationMetrics
    val params:Array[ParamMap] = validatorModel.getEstimatorParamMaps
    // 根据最好的评估结果得到最好的模型和相应的参数
    val bestModel = validatorModel.bestModel
    val bestParams = params(accuracy.indexOf(accuracy.max))
    println("the best params of RandomForest is:",bestParams)
    val forestModel = bestModel.asInstanceOf[PipelineModel].stages(2).asInstanceOf[RandomForestClassificationModel]
    //println(s"the forestModel's numTrees is: ${forestModel.numTrees}" ) //参数输出与bestParams一致
    println("the importance of features in Random Forest is:")
    forestModel.featureImportances.toArray.zip(train.columns).sortBy(_._1).reverse.foreach(println)

    /*3.模型预测*/
    // 在最终测试集上进行预测，得到测试准确率
    val predictions = bestModel.transform(test)
    val testAccuracy = evaluator.evaluate(predictions)
    println(s"the best RF model's accuracy on testDF is: $testAccuracy")
  }

  /*计算样本数据中各类别的比例*/
  def classProbabilities(data:DataFrame, column:String="Cover_Type"):Array[Double]={
    val total = data.count()
    val resultArr = data.groupBy(column)
      .count()
      .orderBy(column)
      .withColumn("count",$"count".cast("Double"))
      .select("count")
      .collect()
      .map(x=>x.getDouble(0)/total)
    resultArr
  }
  /*计算做随机猜测的准确率*/
  def randomClassifier(trainData:DataFrame, testData:DataFrame):Double={
    /*训练集中各类别所占比例，按类别排序*/
    val trainClassProb = classProbabilities(trainData)
    /*测试集中各类别所占比例，按类别排序*/
    val testClassProb = classProbabilities(testData)
    /*在这种“瞎蒙”状态下，预测正确的总准确度可达到多少，zip方法可将对应元素组成pair序列*/
    val accuracy = testClassProb.zip(trainClassProb).map{
      case (testPrior,trainPrior) => testPrior*trainPrior
    }.sum
    accuracy
  }
}

/*建立新的自定义Transformer，将指定列由String转为Double*/
class ColFormatConverter(override val uid:String) extends UnaryTransformer[String, Double, ColFormatConverter]{
  def this() = this(Identifiable.randomUID("ColFormatConverter"))
  override protected def createTransformFunc: String => Double = {
    x:String => x.toDouble
  }
  override protected def validateInputType(inputType: DataType): Unit = {
    require(inputType == StringType)
  }
  override protected def outputDataType: DataType = DoubleType
}



