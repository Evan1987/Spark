
package cigar.Chap07_GraphX

import cigar.Constant._
import scala.xml._
import java.io.{File, PrintWriter}
import org.apache.log4j.{Level, Logger}

case class DocInfo(id: String, topics: Seq[String])
object GraphX {

  val filePath = constant.PATH + "ch07-graph\\data\\"
  val meshTablePath = filePath + "mesh.txt"
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)

    if(!new File(meshTablePath).exists()){
      println("Generate MeshTable!")
      generateMeshTable(filePath, meshTablePath)
    }else{
      println("Target MeshTable already exists!")
    }
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

}


