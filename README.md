<!DOCTYPE html>
<html>
<head>

</head>
<body>


<h2>SessionRecGraphFusion framework</h2>

<h3>Introduction</h3>
<p align="justify">This reproducibility package was prepared for the paper titled "Performance Comparison of Session-based Recommendation Algorithms based on Graph Neural Networks" and submitted to ECIR '24. 
The results reported in this paper were achieved with the help of the SessionRecGraphFusion framework, which is built on the session-rec framework. Session-rec is a 
Python-based framework for building and evaluating recommender systems. It implements a suite of state-of-the-art algorithms and baselines for session-based and 
session-aware recommendation. More information about the session-rec framework can be <a href="https://rn5l.github.io/session-rec/index.html">found here.</a></p>
<h5>The following session-based algorithms have been addded to the session-rec framework. The extended framework is named as SessionRecGraphFusion framework.</h5>
<ul>
  <li>GCE-GNN: Global Context Enhanced Graph Neural Networks for Session-based Recommendation [SIGIR '20]<a href="https://github.com/CCIIPLab/GCE-GNN">(Original Code)</a></li>
  <li>TAGNN: Target Attentive Graph Neural Networks for Session-based Recommendation [SIGIR '20] <a href="https://github.com/CRIPAC-DIG/TAGNN">(Original Code)</a></li>
  <li>MGS: An Attribute-Driven Mirror Graph Network for Session-based Recommendation [SIGIR '22] <a href="https://github.com/WHUIR/MGS">(Original Code)</a></li>
  <li>GNRRW: Graph Neighborhood Routing and Random Walk for Session-based Recommendation [ICDM '21] <a href="https://github.com/resistzzz/GNRRW">(Original Code)</a></li>
  <li>COTREC: Self-Supervised Graph Co-Training for Session-based Recommendation [CIKM '21] <a href="https://github.com/xiaxin1998/COTREC">(Original Code)</a></li>
  <li>FLCSP: Fusion of Latent Categorical Prediction and Sequential Prediction for Session-based Recommendation [Information Sciences (IF: 5.524) Elsevier '21] <a href="https://github.com/RecSysEvaluation/extended-session-rec/tree/master/algorithms/FLCSP">(Original Code)</a></li>
  
  <li>CMHGNN: Category-aware Multi-relation Heterogeneous Graph Neural Networks for Session-based Recommendation [Neurocomputing (IF: 5.719) Elsevier '20] <a href="https://github.com/RecSysEvaluation/extended-session-rec/tree/master/algorithms/CM_HGCN">(Original Code)</a></li>
</ul>
<h5>Required libraries to run the framework</h5>
<ul>
  <li>Anaconda 4.X (Python 3.5 or higher)</li>
  <li>numpy=1.23.5</li>
  <li>pandas=1.5.3</li>
  <li>torch=1.13.1</li>
  <li>scipy=1.10.1</li>
  <li>python-dateutil=2.8.1</li>
  <li>pytz=2021.1</li>
  <li>certifi=2020.12.5</li>
  <li>pyyaml=5.4.1</li>
  <li>networkx=2.5.1</li>
  <li>scikit-learn=0.24.2</li>
  <li>keras=2.11.0</li>
  <li>six=1.15.0</li>
  <li>theano=1.0.3</li>
  <li>psutil=5.8.0</li>
  <li>pympler=0.9</li>
  <li>Scikit-optimize</li>
  <li>tensorflow=2.11.0</li>
  <li>tables=3.8.0</li>
  <li>scikit-optimize=0.8.1</li>
  <li>python-telegram-bot=13.5</li>
  <li>tqdm=4.64.1</li>
  <li>dill=0.3.6</li>
  <li>numba</li>
</ul>
<h2>Installation guide</h2>  
<p>This is how the framework can be downloaded and configured to run the experiments</p>
  
<h5>Using Docker</h5>
<ul>
  <li>Download and install Docker from <a href="https://www.docker.com/">https://www.docker.com/</a></li>
  <li>Run the following command to "pull Docker Image" from Docker Hub: <code>docker pull shefai/session_rec_graph_fusion:latest</code>
  <li>Clone the GitHub repository by using the link: <code>https://github.com/Faisalse/SessionRecGraphFusion.git</code>
  <li>Move into the <b>SessionRecGraphFusion</b> directory</li>
  
  <li>Run the command to mount the current directory <i>SessionRecGraphFusion</i> to the docker container named as <i>session_rec_graph_container</i>: <code>docker run --name session_rec_graph_container  -it -v "$(pwd):/SessionRecGraphFusion" -it shefai/session_rec_graph_fusion:latest</code>. If you have the support of CUDA-capable GPUs then run the following command to attach GPUs with the container: <code>docker run --name session_rec_graph_container  -it --gpus all -v "$(pwd):/SessionRecGraphFusion" -it shefai/session_rec_graph_fusion:latest</code></li> 
<li>If you are already inside the runing container then run the command to navigate to the mounted directory <i>SessionRecGraphFusion</i>: <code>cd /SessionRecGraphFusion</code> otherwise starts the "session_rec_graph_container" and then run the command</li>
<li>Finally run this command to reproduce the results: <code>python run_config.py conf/in conf/out</code></li>
</ul>  

  
<h5>Using Anaconda</h5>
  <ul>
    <li>Download Anaconda from <a href="https://www.anaconda.com/">https://www.anaconda.com/</a> and install it</li>
    <li>Clone the GitHub repository by using this link: <code>https://github.com/Faisalse/SessionRecGraphFusion.git</code></li>
    <li>Open the Anaconda command prompt</li>
    <li>Move into the <b>SessionRecGraphFusion</b> directory</li>
    <li>Run this command to create virtual environment: <code>conda create --name SessionRecGraphFusion python=3.8</code></li>
    <li>Run this command to activate the virtual environment: <code>conda activate SessionRecGraphFusion</code></li>
    <li>Run this command to install the required libraries for CPU: <code>pip install -r requirements_cpu.txt</code>. However, if you have support of CUDA-capable GPUs, 
        then run this command to install the required libraries to run the experiments on GPU: <code>pip install -r requirements_gpu.txt</code></li>
    <li>Finally run this command to reproduce the results: <code>python run_config.py conf/in conf/out</code></li>
    <li>If you do not understand the instructions, then check the video to run the experiments: (https://youtu.be/uCW2omAxYP8?si=UW_YjJ_GqACuc_Gs)</li>
  </ul>
  <p align="justify">In this study, we use the <a href="https://competitions.codalab.org/competitions/11161#learn_the_details-data2">DIGI</a>, <a href="https://www.kaggle.com/datasets/chadgostopp/recsys-challenge-2015">RSC15</a> and <a href="https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset">RETAIL</a> datasets to evaluate the performance of recently
published GNN models and to check how they react to the different values of embedding sizes and random seeds. We also conduct the experiments related to the tuning of models on the test data instead of validation data. The reproducibility files to run the experiments can be found in the <i>conf folder</i>. If you want to run the experiments, copy the configuration file from the <i>conf folder</i> and paste it into the <i>in folder</i>, and again run the command <code>python run_config.py conf/in conf/out</code> to reproduce the results. 
</p>



</body>
</html>  

