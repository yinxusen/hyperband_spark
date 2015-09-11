name := "hyperband"

version := "0.1"

scalaVersion := "2.10.4"

resolvers ++= Seq(
  "Typesafe Repository" at "http://repo.typesafe.com/typesafe/releases/",
  "Sonatype Snapshots 1" at "https://oss.sonatype.org/content/repositories/snapshots/",
  "Sonatype Snapshots 2" at "https://oss.sonatype.org/content/repositories/releases/"
)

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core"    % "1.4.1",
  "org.apache.spark" %% "spark-mllib"   % "1.4.1",
  "org.apache.spark" %% "spark-graphx" % "1.4.1",
  "org.apache.spark" %% "spark-sql" % "1.4.1",
  "org.scalatest" %% "scalatest" % "2.2.1" % "test"
)
