# 实验三 Git使用实验报告
序号：06  姓名：李天明  学号：20221200703 
## 一、实验目的

通过实际操作，深入理解Git版本控制系统的基本概念、工作原理以及常用操作，掌握在Windows环境下安装Git、创建版本库、进行版本控制、远程仓库操作、分支管理以及多人协作等技能，为后续的软件开发项目提供版本管理支持。

## 二、实验环境

- 操作系统：Windows 11
    
- Git版本：2.38.1
    
- 远程仓库：GitHub
    

## 三、实验内容与步骤

### （一）Git的安装与配置

1. 从Git官网下载Windows版Git安装包并进行默认安装。
    
2. 安装完成后，在开始菜单中找到“Git → Git Bash”，打开命令行窗口，验证安装是否成功。
    
3. 在命令行中输入以下命令，设置全局用户名和邮箱：
    
    ```
    git config --global user.name "ApoLee"
    git config --global user.email "Haodim@outlook.com"
    ```
    ![[set_user_info.png]]
    

### （二）创建版本库

1. 在D盘下新建一个名为“git_test”的文件夹。
2. 执行`git init`命令，将该目录初始化为Git可以管理的仓库，此时目录下会多出一个隐藏的`.git`目录。
    ![[cd_D_git_init.png]]

### （三）基本操作

1. 在“git_test”目录下新建一个名为“readme.txt”的文本文件，内容为“a”。
    
2. 使用`git add readme.txt`命令将文件添加到暂存区。
    
3. 执行`git commit -m " commit"`命令，将文件提交到仓库。
    ![[commit readme a.png]]
4. 修改“readme.txt”文件，在第二行添加内容“a”，然后使用`git status`查看状态。
    
5. 使用`git diff readme.txt`查看文件的修改内容。
    
6. 再次执行`git add readme.txt`和`git commit -m "Add line 2"`命令，将修改提交到仓库。
    ![[Add line2.png]]

### （四）版本回退

1. 继续修改“readme.txt”文件，在第三行添加内容“c”，提交修改。
    
2. 使用`git log`查看提交历史，记录下最近三次提交的版本号。
    ![[Cat log.png]]
3. 执行`git reset --hard HEAD^`命令，将版本回退到上一个版本，使用`cat readme.txt`查看文件内容，确认回退成功。
    
4. 使用`git reflog`查看版本号，找到之前添加“c”内容的版本号。
    
5. 使用`git reset --hard [版本号]`命令，将版本恢复到添加“c”内容的版本。
    ![[reset hard.png]]

### （五）工作区与暂存区的区别

1. 在“readme.txt”文件中再添加一行内容“d”，同时在目录下新建一个文件“test.txt”，内容为“test”。
    
2. 使用`git status`查看状态，观察工作区和暂存区的差异。
    
3. 执行`git add readme.txt test.txt`命令，将两个文件添加到暂存区。
    
4. 再次使用`git status`查看状态，确认文件已添加到暂存区。
    
5. 执行`git commit -m "Add file test.txt"`命令，将暂存区的内容提交到分支上。
    ![[add new file test.txt.png]]

### （六）撤销修改和删除文件操作

1. 在“readme.txt”文件中添加一行内容“e”，使用`git status`查看状态。
    
2. 执行`git restore readme.txt`命令，撤销工作区对“readme.txt”文件的修改。
    ![[restore_1.png]]
3. 在“git_test”目录下新建一个文件“AAA.txt”，内容为“A”，提交到仓库。
    
4. 使用`rm AAA.txt`命令删除文件“AAA.txt”，然后执行`git status`查看状态。
    
5. 执行`git restore -- AAA.txt`命令，恢复文件“b.txt”到工作区。
    ![[restore file.png]]
![[val restore file.png]]
### （七）远程仓库操作

1. 在GitHub上注册账号并登录。
    
2. 创建一个新的仓库“git_test”，记录下仓库的HTTPS地址。
    
3. 在本地“git_test”仓库目录下，执行以下命令添加远程仓库：
    
    `git remote add origin [仓库的HTTPS地址]`
    ![[连接github仓库.png]]
4. 执行`git push -u origin master`命令，将本地仓库的内容推送到远程仓库。
    ***tips****：这里遇到一个小问题：
    为了解决这个问题，重新配置了网络端口操作如下
    - 检查并确认系统的代理端口（经查阅为7890）。
	- 配置 Git 使用该代理端口。
	- 刷新 DNS 缓存以确保网络连接通畅。![[网络配置.png]]![[push repos.png]]
5. 在GitHub页面上查看远程仓库的内容，确认推送成功。
    ![[GitHub仓库1.png]]
    

### （八）分支管理

1. 在本地“git_test”仓库目录下，执行`git branch dev`命令创建一个名为“dev”的分支。
    
2. 执行`git checkout dev`命令切换到“dev”分支。
    ![[create branch.png]]
3. 在“dev”分支上修改“readme.txt”文件，在最后一行添加内容“F”，提交修改。
    
4. 执行`git checkout master`命令切换回“master”分支，查看“readme.txt”文件的内容。
    
5. 在“master”分支上执行`git merge dev`命令，将“dev”分支的内容合并到“master”分支。
    
6. 查看合并后的“readme.txt”文件内容，确认合并成功。
    ![[merge.png]]
7. 使用`git log`命令查看分支合并的情况。
    

### （九）多人协作
（同上操作）
1. 在GitHub上创建一个新的仓库“testgit3”，记录下仓库的HTTPS地址。
    
2. 在本地克隆该仓库：
    
    `git clone [仓库的HTTPS地址]`
    
3. 进入克隆后的本地仓库目录，创建一个新的分支“dev”，并在该分支上进行开发，修改“readme.txt”文件，提交修改。
    
4. 执行`git push origin dev`命令，将本地“dev”分支的内容推送到远程仓库。
    
5. 模拟另一个开发者，在另一台电脑上（或同一台电脑的不同目录下）克隆远程仓库，创建本地“dev”分支并进行开发，修改“readme.txt”文件，提交修改。
    
6. 尝试将本地修改推送到远程仓库，如果出现冲突，按照提示使用`git pull`命令合并远程分支的最新提交，解决冲突后再次提交并推送。
    

## 四、实验结果

通过本次实验，成功在Windows环境下安装并配置了Git，创建了本地版本库，并进行了文件的添加、提交、版本回退、撤销修改、删除文件等基本操作。掌握了工作区与暂存区的区别，能够熟练地进行远程仓库的添加、推送和克隆操作。学会了分支的创建、切换、合并以及冲突解决方法，并体验了多人协作开发的流程。实验过程中，所有操作均按照预期完成，本地仓库与远程仓库的内容保持一致，分支管理策略得到合理应用，多人协作中的冲突能够有效解决。

## 五、实验总结

本次实验让我对Git版本控制系统有了更深入的理解和实践经验。Git的强大功能和灵活性为软件开发提供了有力的版本管理支持，尤其是在多人协作开发中，能够有效避免文件冲突和版本混乱的问题。通过实验，我掌握了Git的基本操作和常用命令，提高了对版本控制的认识和操作能力。在今后的软件开发项目中，我将能够更好地利用Git进行代码管理和团队协作，提高开发效率和代码质量。同时，我也意识到在实际开发过程中，还需要进一步学习和掌握更高级的Git功能和技巧，以应对更复杂的开发场景和需求。