#! /bin/bash

# To use this file, you may wanna UNCOMMENT some of the command for the first use.

# For ubuntu information: show ubuntu version command:

lsb_release -a

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
echo "alias"" ""v=\"vim\"" >> .bashrc
echo "alias"" ""gis=\"git status\"" >> .bashrc
echo "alias"" ""gicm=\"git commit -m\"" >> .bashrc
echo "alias"" ""gips=\"git push\"" >> .bashrc
echo "alias"" ""gipl=\"git pull\"" >> .bashrc
echo "alias"" ""gia=\"git add\"" >> .bashrc
echo "alias"" ""py=\"python3\"" >> .bashrc
echo "alias"" ""rb=\"reboot\"" >> .bashrc
echo "alias"" ""cdd=\"cd \/media\/wb\/TOSHIBA\\ EXT\/2\/d\/dataguru\"" >> .bashrc

echo "alias"" ""cdd2=\"cd /media/wb/TOSHIBA\ EXT/2/d/dataguru/ \"" >> .bashrc

# Then you can install vim

sudo apt-get install vim

# Install unrar

sudo apt install unrar
# Please uncomment the following lines to install anaconda.

# Then, to install anaconda. The fastest way to download is using Thunderstorm on Windows Platform, which takes about 3 - 5 mins.

# Prerequisites for installing anaconda, which takes about 20 seconds:

# sudo apt-get install libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6

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


# Uncomment the following lines to install VLC Player.

# Install VLC Player for playing mp4

# sudo add-apt-repository ppa:videolan/master-daily
# sudo apt update

# To install:

# sudo apt install vlc qtwayland5

#In order to use the streaming and transcode features in VLC for Ubuntu 18.04, enter the following command to install the libavcodec-extra packages.

# sudo apt install libavcodec-extra

# set up drivers for GPU:
# GO TO https://www.nvidia.com/Download/index.aspx?lang=en-us to download the shell script for drivers. Around 100 MB. Then execute the script.

# If installation is failed due to current installed graphics driver, you may want to remove the current driver. For instructions, go do the directory in the current directory here, following the .txt file.

# Now, the VLC plaer and the graphics driver have been installed, you could use VLC player to play the mp4 file. In case there's no sound, it may due to the setup in linux. Change the output in the linux sound settings to HDMI rather than S/FPDI if you are using HDMI.

# To set the VLC as the default player, simpler search for "default"
