#! /bin/bash

# The very first step of running this file is to GET this file...... So you may need to just copy this file or clone the whole repository with git clone. Steps are the following:

# Use sudo apt-get update for updating the apt-get
sudo apt-get update

# Install git before cloning

sudo apt-get install git

# Change directory to the home directory

cd

# Clone git remot repository:

git clone https://github.com/wubinbai/2019.git


# 



# First you may wanna follow the instructions on the welcome page. Options are to set up the livepatch: U.s.e.R. : wubinbai e.m.a.i.1: wubinbai@yahoo.com p.s.w: b**00****

# Then setup font manually. Open a terminal,  go to the preference and text, set the font size to around 42-46.

# You may want to alias sd as shutdown now to fasten your shutdown using two letters:

# Go to the ~/ first:

cd
echo "# customized aliases by wubin" >> .bashrc
echo "alias"" ""sd=\"shutdown now\"" >> .bashrc
echo "alias"" ""ipy=\"ipython3\"" >> .bashrc




# Then you can install vim

sudo apt-get install vim



# Then, to install anaconda. The fastest way to download is using Thunderstorm on Windows Platform, which takes about 3 - 5 mins.

# Prerequisites for installing, which takes about 20 seconds:

sudo apt-get install libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6

# Then, go to the anaconda.sh you have downloaded, and run the ./xxx.sh

# The installer prompts “In order to continue the installation process, please review the license agreement.” PRESS ENTER to view license terms.

# Scroll to the bottom, ENTER "yes"

# The installer prompts you to PRESS ENTER to accept the default install location. # We recommend you accept the default install location. Do not choose the path as /usr for the Anaconda/Miniconda installation.

# The installer prompts “Do you wish the installer to initialize Anaconda3 by running conda init?” We RECOMMEND “yes”.

# Then, all done. Anaconda and JetBrains are working together to bring you Anaconda-powered environments tightly integrated in the PyCharm IDE.

# Close and open your terminal window for the installation to take effect, OR you can enter the command source ~/.bashrc.

# To control whether or not each shell session has the base environment activated or not, run conda config --set auto_activate_base False or True. To run conda from anywhere without having the base environment activated by default, use conda config --set auto_activate_base False. This only works if you have run conda init first.

# After your install is complete, verify it by opening Anaconda Navigator, a program that is included with Anaconda: Open a terminal window and type anaconda-navigator. If Navigator opens, you have successfully installed Anaconda. If not, check that you completed each step above, then see our Help page.

# After installing anaconda, you have already installed a lot of related packages, like jupyter notebook!!!

# You have now installed ipython, and you may wanna configure it, since you have cloned the repository, you can simply copy the file in the repository into the directory where the ipython import packages. The following line does this.

cp ~/2019/Config_ipython_import/ipython/import_here.py ~/.ipython/profile_default/startup/


