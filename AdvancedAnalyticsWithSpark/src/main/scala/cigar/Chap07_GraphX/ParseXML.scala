
package cigar.Chap07_GraphX

import scala.xml._

object ParseXML {
  /**
    * 获取文章 MeSH标签
    * return: (文章ID， 主题列表)
    * */
  def getTopicsOfDoc(docNode: Node): DocInfo = {
    val id = (docNode \ "PMID").text
    val descriptorNames = docNode \\ "DescriptorName"  //直接获取DescriptorName条目
    val topicElems = descriptorNames.filter(node => (node \ "@MajorTopicYN").text == "Y")  // 查询直接子节点的MajorTopicYN值
    val topics = topicElems.map(_.text)
    DocInfo(id, topics)
  }

  /**
    * 从单一文件中获取全部文章信息
    * */
  def getDocTopicFromFile(fileElem: Elem): Seq[DocInfo] = {
    val docs = fileElem \ "MedlineCitation"  // 获取文章条目
    docs.map(docNode => getTopicsOfDoc(docNode))
  }
}
