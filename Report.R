###################################
##title: "Classification Report"###
##  Student Number: gfjc92  #######
###################################


##install package
install.packages("readr")
install.packages("skimr")
install.packages("mlr3verse")
install.packages("ranger")
install.packages("ggplot2")
install.packages("DataExplore")


##I: Problem description

#read the csv

library("readr")
bank_loan <- readr::read_csv("https://www.louisaslett.com/Courses/MISCADA/bank_personal_loan.csv")
head(bank_loan)

set.seed(53) # set fixed seed

##We need to duel with some datalines, turn some datalines as factors. And ZIP.Code in the table is a code of home adress, we just take the first two character as the value.


bank_loan$Personal.Loan<-factor(bank_loan$Personal.Loan) #change the form "numeric" to "factor"
bank_loan$Education<-factor(bank_loan$Education)
bank_loan$Securities.Account<-factor(bank_loan$Securities.Account)
bank_loan$CD.Account<-factor(bank_loan$CD.Account)
bank_loan$Online<-factor(bank_loan$Online)
bank_loan$CreditCard<-factor(bank_loan$CreditCard)
bank_loan$ZIP.Code<-bank_loan$ZIP.Code/1000     #catch the first two number of the code as the value
bank_loan$ZIP.Code<-floor(bank_loan$ZIP.Code)

head(bank_loan)     #show the head of the data and check the form of every lines.


#show basic data states

library("skimr")
bank_loan<-as.data.frame(bank_loan)
skimr::skim(bank_loan)


##plot basic images

library("DataExplorer")
DataExplorer::plot_bar(bank_loan, ncol = 3)
DataExplorer::plot_histogram(bank_loan, ncol = 3)
DataExplorer::plot_boxplot(bank_loan, by = "Personal.Loan", ncol = 3)

##II Model fitting

##load packages

library("data.table")
library("mlr3verse")
library("ranger")
library("ggplot2")




#classification task setting
bank_task <- TaskClassif$new(id = "bank_loan",
                             backend = bank_loan, 
                             target = "Personal.Loan",
                             positive = "0")
bank_task


#use cross validation for resampling

cv5 <- rsmp("cv", folds = 5)
cv5$instantiate(bank_task)

#setting model
lrn_baseline <- lrn("classif.featureless", predict_type = "prob")
lrn_cart <- lrn("classif.rpart", predict_type = "prob")
lrn_log <- lrn("classif.log_reg", predict_type = "prob")
lrn_ran <- lrn("classif.ranger", predict_type = "prob",importance="permutation")


#resample
res_baseline <- resample(bank_task, lrn_baseline, cv5, store_models = TRUE)
res_cart <- resample(bank_task, lrn_cart, cv5, store_models = TRUE)
res_log <- resample(bank_task, lrn_log, cv5, store_models = TRUE)
res_ran <- resample(bank_task, lrn_ran, cv5, store_models = TRUE)

# Look at accuracy
res_baseline$aggregate()
res_cart$aggregate()
res_log$aggregate()
res_ran$aggregate()


#compare the result by benchmark

res <- benchmark(data.table(
  task       = list(bank_task),
  learner    = list(lrn_baseline,
                    lrn_cart,
                    lrn_log,
                    lrn_ran),
  resampling = list(cv5)
), store_models = TRUE)


print(res$aggregate(list(msr("classif.ce"),
                   msr("classif.acc"),
                   msr("classif.auc"),
                   msr("classif.fpr"),
                   msr("classif.fnr"))))


##the predictions and confusion matrixs

pred_baseline=res_baseline$prediction()
print(pred_baseline$confusion)
pred_cart=res_cart$prediction()
print(pred_cart$confusion)
pred_log=res_log$prediction()
print(pred_log$confusion)
pred_ran=res_ran$prediction()
print(pred_ran$confusion)


##III Model improvements

#use random forest to compare the importance of variables

part_bank_task=mlr3::partition(bank_task,ratio=0.67,stratify=TRUE)
lrn_ran$train(bank_task, row_ids = part_bank_task$train)

lrn_ran$importance()


bank_importance = as.data.table(lrn_ran$importance(), keep.rownames = TRUE)
lrn_ran$importance()
colnames(bank_importance) = c("Feature", "Importance")
ggplot(data=bank_importance,
       aes(x = reorder(Feature, Importance), y = Importance)) + 
  geom_col() + coord_flip() + xlab("")

##choose seven the most important variables and do the model fitting again, trying to check whether the fitting result will be influenced


improve_bank_loan<-bank_loan
improve_bank_loan$Mortgage<-NULL
improve_bank_loan$CreditCard<-NULL
improve_bank_loan$Online<-NULL
improve_bank_loan$Securities.Account<-NULL
improve_bank_loan$ZIP.Code<-NULL
head(improve_bank_loan)

im_bank_task <- TaskClassif$new(id = "im_bank_loan",
                                backend = improve_bank_loan, 
                                target = "Personal.Loan",
                                positive = "0")
im_bank_task 

im_cv5 <- rsmp("cv", folds = 5)
im_cv5$instantiate(im_bank_task)

im_lrn_baseline <- lrn("classif.featureless", predict_type = "prob")
im_lrn_cart <- lrn("classif.rpart", predict_type = "prob")
im_lrn_log <- lrn("classif.log_reg", predict_type = "prob")
im_lrn_ran <- lrn("classif.ranger", predict_type = "prob")

im_res_baseline <- resample(im_bank_task, im_lrn_baseline, im_cv5, store_models = TRUE)
im_res_cart <- resample(im_bank_task, im_lrn_cart, im_cv5, store_models = TRUE)
im_res_log <- resample(im_bank_task, im_lrn_log, im_cv5, store_models = TRUE)
im_res_ran <- resample(im_bank_task, im_lrn_ran, im_cv5, store_models = TRUE)

im_res <- benchmark(data.table(
  task       = list(im_bank_task),
  learner    = list(im_lrn_baseline,
                    im_lrn_cart,
                    im_lrn_log,
                    im_lrn_ran),
  resampling = list(im_cv5)
), store_models = TRUE)
im_res
print(im_res$aggregate(list(msr("classif.ce"),
                      msr("classif.acc"),
                      msr("classif.auc"),
                      msr("classif.fpr"),
                      msr("classif.fnr"))))

im_pred_baseline=im_res_baseline$prediction()
print(im_pred_baseline$confusion)
im_pred_cart=im_res_cart$prediction()
print(im_pred_cart$confusion)
im_pred_log=im_res_log$prediction()
print(im_pred_log$confusion)
im_pred_ran=im_res_ran$prediction()
print(im_pred_ran$confusion)
