@Library(['github.com/WORSICA/jenkins-pipeline-library@indigo-dependencies']) _

def projectConfig

pipeline {
    agent any

    options {
        buildDiscarder(logRotator(daysToKeepStr: '7', numToKeepStr: '1'))
    }

    stages {
        
        stage('SQA Baseline Dynamic Stages') {
            steps {
                script {
                    projectConfig = pipelineConfig()
                    buildStages(projectConfig)
                }
            }
            post {
                cleanup {
                    cleanWs()
                }
            }
        }

    }

}
