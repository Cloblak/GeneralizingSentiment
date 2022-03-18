from google.colab import drive
drive.mount('/content/drive')


# Remove Colab default sample_data
!rm -r ./sample_data

!ls


import os
os.getcwd()

# pulling in github
import os
import getpass
!curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
!sudo apt-get install git-lfs
!git lfs install

# Remove Colab default sample_data
!rm -r ./sample_data

# Clone GitHub files to colab workspace
git_user = "Cloblak" # Enter user or organization name
git_token = "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" # Enter your email
repo_name = "Cloblak/GeneralizingSentiment" # Enter repo name
git_path = f"https://{git_user}:{git_token}@github.com/{repo_name}.git"
!git clone "{git_path}"


# run streamlit
!pip install streamlit
!pip install pyngrok
!ngrok authtoken 26Tk57POzy7LjVySmB9tr6bCakE_3jBEmhF92UYHJtiBamauh


import os
os.chdir("drive/MyDrive/GenSentiment/GeneralizingSentiment/scripts")
!ls



!wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip


!unzip ngrok-stable-linux-amd64.zip


get_ipython().system_raw('./ngrok http 8501 &')


!curl -s http://localhost:4040/api/tunnels | python3 -c \
    'import sys, json; print("Execute the next cell and the go to the following URL: " +json.load(sys.stdin)["tunnels"][0]["public_url"])'


# !streamlit run --server.port 8051 app.py &>/dev/null&


!streamlit run app.py


!pip install transformers
!pip install SentencePiece

!ngrok authtoken 26Tk57POzy7LjVySmB9tr6bCakE_3jBEmhF92UYHJtiBamauh
!ngrok.connect(8051)
!streamlit run --server.port 8051 app.py &>/dev/null&



from pyngrok import ngrok


# Setup a tunnel to the streamlit port 8501
public_url = ngrok.connect(port="8051")
public_url



!pgrep streamlit


!kill 2710



ngrok.kill()













