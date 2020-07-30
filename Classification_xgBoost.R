setwd("//ahmct-065/teams/PMRF/Amir/")

library(data.table)
library(dplyr)
library(tidyr)
library(caret)
library(anytime)
library(e1071)
library(DMwR)
library(glmnet)
library(doParallel)

library(xgboost)
library(DiagrammeR)
library(ggplot2)
library(hrbrthemes)
library(viridis)
library(ggrepel)
library(SHAPforxgboost)
library(forcats)
library(Boruta)
library(reshape2)


library(Ckmeans.1d.dp)
library(devtools)
library(xgboostExplainer)


set.seed(123)

df=fread(file="./bin/LEMO_CHP.by.roadCond_workOrderDate.csv", sep=",", header=TRUE)
df[df==""]=NA

#select features
colnames(df)
selected_cols=c("work_date", "activity", "district", "county", "route", "work_duration", "work_length", 
                "closure_id", "closure_coverage", "closure_length", "closure_workType", "closure_duration", "closure_cozeepMazeep", 
                "closure_detour", "closure_type", "closure_facility", "closure_lanes",
                "surface_type", "num_lanes", "road_use", "road_width", "median_type", "barrier_type", "hwy_group", "access_type", 
                "terrain_type", "road_speed", "road_adt", "population_code", "peak_aadt", "aadt", "truck_aadt", "collision_density11_12", "collision_id", 
                "collision_time", "collision_day", "collision_weather_cond_1", "collision_weather_cond_2", "collision_location_type", 
                "collision_ramp_intersection", "collision_severity", "collision_num_killed", "collision_num_injured", "collision_party_count", 
                "collision_prime_factor", "collision_violation_cat", "collision_surface_cond", "collision_road_cond_1", "collision_road_cond_2", 
                "collision_lighting_cond", "collision_control_device", "collision_road_type")

#clean up the selected features and convert to type
source("./Codes/FUNC(Dataset CleanUp).R")
df=cleanUp_Dataset(df, selected_cols)

#check clean up process
df %>% str

#filter rows for a complete data set, in that, no features except collision and closure features should be missing
df=na.omit(setDT(df), cols = c("work_month", "work_day", "district", "county", "route", "activity", "work_duration", "work_length", 
                               "surface_type", "num_lanes", "road_use", "road_width", "median_type", "barrier_type", "hwy_group", 
                               "access_type", "terrain_type", "road_speed", "road_adt", "population_code", 
                               "peak_aadt", "aadt", "truck_aadt", "collision_density11_12"))

#create training and testing data set
train.ind=createDataPartition(df$collision_id, times = 1, p=0.7, list = FALSE)
training.df=df[train.ind, ]
testing.df=df[-train.ind, ]

#check and plot response variable class
length(which(training.df$collision_id==1))/length(training.df$collision_id)
ggplot(data=training.df, aes(x=collision_id, fill=collision_id))+
  geom_bar()+
  theme(axis.text.x = element_text(angle = 0, hjust = 0.5, size=14),
        axis.title.x = element_text(size = 20, face="bold"),
        axis.text.y = element_text(size=14),
        axis.title.y = element_text(size=20, face = "bold"), legend.position = "none")+
  ylab("Count")+
  xlab("Class collision")
###############################################################################################################################################
###############################################################################################################################################
###############################################################################################################################################
#drop collision and closure columns, some of NA variabels can be translated to 0-1 categories or numerics
temp.df=training.df[,-c("closure_workType", "closure_duration", "closure_type", "closure_facility")]
temp.df$closure_cozeepMazeep=ifelse(is.na(temp.df$closure_cozeepMazeep), 0, 1)
temp.df$closure_detour=ifelse(is.na(temp.df$closure_detour), 0, 1)

temp.df=temp.df[,-c("closure_lanes")]
temp.df$closure_coverage[is.na(temp.df$closure_coverage)]=0
temp.df$closure_coverage=abs(temp.df$closure_coverage)
temp.df$closure_length[is.na(temp.df$closure_length)]=0

temp.df=temp.df[,-c("collision_time", "collision_day", "collision_weather_cond_1", "collision_weather_cond_2", 
                                    "collision_location_type", "collision_ramp_intersection", "collision_severity", "collision_prime_factor", 
                                    "collision_violation_cat", "collision_surface_cond", "collision_road_cond_1", "collision_road_cond_2", 
                                    "collision_lighting_cond", "collision_control_device", "collision_road_type")]

temp.df=temp.df[,-c("collision_num_killed", "collision_num_injured", "collision_party_count")]

test.df=testing.df[,-c("closure_workType", "closure_duration", "closure_type", "closure_facility")]
test.df$closure_cozeepMazeep=ifelse(is.na(test.df$closure_cozeepMazeep), 0, 1)
test.df$closure_detour=ifelse(is.na(test.df$closure_detour), 0, 1)

test.df=test.df[,-c("closure_lanes")]
test.df$closure_coverage[is.na(test.df$closure_coverage)]=0
test.df$closure_coverage=abs(test.df$closure_coverage)
test.df$closure_length[is.na(test.df$closure_length)]=0

test.df=test.df[,-c("collision_time", "collision_day", "collision_weather_cond_1", "collision_weather_cond_2", 
                    "collision_location_type", "collision_ramp_intersection", "collision_severity", "collision_prime_factor", 
                    "collision_violation_cat", "collision_surface_cond", "collision_road_cond_1", "collision_road_cond_2", 
                    "collision_lighting_cond", "collision_control_device", "collision_road_type")]

test.df=test.df[,-c("collision_num_killed", "collision_num_injured", "collision_party_count")]
###############################################################################################################################################
###############################################################################################################################################
###############################################################################################################################################
#prepare sparse matrices for xgboost
preprocess.mod=preProcess(temp.df, method = c("center", "scale"), rangeBounds = c(0,1))
temp.df=predict(preprocess.mod, temp.df)
test.df=predict(preprocess.mod, test.df)

dtest=sparse.model.matrix(collision_id~.-1, data = data.frame(test.df))
dtrain=sparse.model.matrix(collision_id~.-1, temp.df)
label=as.numeric(as.character(training.df$collision_id))

#evaluate the weight of each class in response variable
sumwpos=sum(label==1)
sumwneg=sum(label==0)

##############################################################
#########################################parameter hypertuning
#register parallel backend
myCl=makeCluster(detectCores()-1, outfile="Log.txt")
registerDoParallel(myCl)

#define the parameter grid
xgb.grid=expand.grid(nrounds=100, 
                     eta=seq(0.1, 2, 0.2),
                     max_depth=seq(3, 10, 1),
                     gamma = seq(0, 10, 2.5), 
                     subsample = 0.1,
                     min_child_weight = c(1, 3, 5), 
                     colsample_bytree = 1)

#define caret training controls
xgb.control=trainControl(method = "cv",
                         number = 5,
                         verboseIter = TRUE,
                         returnData = FALSE,
                         returnResamp = "none",
                         classProbs = TRUE,
                         allowParallel = TRUE)

xgb.train = train(x = dtrain,
                  y = factor(label, 
                             labels = c("No.Collision", "Collision")),
                  trControl = xgb.control,
                  tuneGrid = xgb.grid,
                  method = "xgbTree",
                  scale_pos_weight=sumwneg/sumwpos,
                  tree_method="hist")

#result of the grid search
xgb.train$bestTune
params=list("eta"=xgb.train$bestTune$eta,
            "max_depth"=xgb.train$bestTune$max_depth,
            "gamma"=xgb.train$bestTune$gamma,
            "min_child_weight"=xgb.train$bestTune$min_child_weight,
            "nthread"=4,
            "objective"="binary:logistic")

#run a cross-validated search for the best number of iterations
xgb.crv=xgb.cv(params = params,
               data = dtrain,
               nrounds = 500,
               nfold = 5,
               label = label,
               showsd = TRUE,
               metrics = list("auc", "rmse", "logloss"),
               stratified = TRUE,
               verbose = TRUE,
               print_every_n = 1L,
               early_stopping_rounds = 50,
               scale_pos_weight=sumwneg/sumwpos)

#result of the cross-validation
xgb.crv$best_iteration
#plot of the cross-validated training and test error
ggplot(xgb.crv$evaluation_log, aes(x=iter))+
  geom_line(aes(y=train_auc_mean, 
                color="Training accuracy"), 
            size=1.2)+
  geom_ribbon(aes(y=train_auc_mean, 
                  ymax=train_auc_mean+train_auc_std,
                  ymin=train_auc_mean-train_auc_std,
                  alpha=0.3))+
  geom_line(aes(y=test_auc_mean, 
                color="Testing accuracy"), 
            size=1.2)+
  geom_ribbon(aes(y=train_auc_mean, 
                  ymax=test_auc_mean+test_auc_std,
                  ymin=test_auc_mean-test_auc_std,
                  alpha=0.3))+
  theme_ipsum(axis_title_just = "center")+
  theme(plot.title = element_blank(),
        legend.text = element_text(size = 18, 
                                   family = "Century Gothic", 
                                   color = "black"),
        axis.text.x = element_text(hjust = 0.5, 
                                   size=18, 
                                   family = "Century Gothic", 
                                   color = "black"),
        axis.title.x = element_text(size = 18, 
                                    family = "Century Gothic", 
                                    color = "black", 
                                    margin = margin(15, 0, 0, 0)),
        axis.text.y = element_text(size=18, 
                                   family = "Century Gothic", 
                                   color = "black"),
        axis.title.y = element_text(size=18, 
                                    family = "Century Gothic", 
                                    color = "black",
                                    margin = margin(0, 15, 0, 0)),
        axis.line.x = element_line(size=1.2),
        axis.line = element_line(size=1.2))+
  xlab("Iteration")+ylab("Avg. accuracy")+
  scale_alpha(guide="none")+
  labs(color="")

#run an instance of xgboost to evaluate initial feature importances
xgb.mod=xgboost(data = dtrain, 
                label = label, 
                max.depth=xgb.train$bestTune$max_depth, 
                eta=xgb.train$bestTune$eta, 
                nthread=4, 
                min_child_weight=xgb.train$bestTune$min_child_weight,
                scale_pos_weight=sumwneg/sumwpos, 
                eval_metric="auc", 
                eval_metric="rmse", 
                eval_metric="logloss",
                gamma=xgb.train$bestTune$gamma,
                nrounds=xgb.crv$best_iteration, 
                objective="binary:logistic",
                tree_method="hist",
                lambda=1,
                alpha=1)

#evaluate and plot feature importance
importance=xgb.importance(feature_names = colnames(dtrain), model = xgb.mod)
feat.label=importance$Feature[1:30]
feat.label=c("Closure = 1", "Work length", "Collision density", "Truck AADT",
             "ADT", "Closure length", "Closure coverage", "Work duration", "Peak AADT",
             "AADT", "Design speed", "Route ID = 10", "County = SJ", "Activity code = M90000",
             "Road width", "Work day = Wed.", "Surface type = C", "Work month = Sep.", "Work month = Jul.", 
              "Route ID = 210", "Work day = Fri.", "Work day = Thu.", 
             "Work day = Mon.", "Work day = Tue.", "Barrier type = E", "Work month = Jan.",
             "Work month = Dec.", "District = 8", "District = 4", "Work month = Aug.")

(gg=xgb.ggplot.importance(importance_matrix = importance[1:30,]))
gg+theme_ipsum(axis_title_just = "center")+
  theme(plot.title = element_blank(),
         axis.text.x = element_text(hjust = 0.5, 
                                    size=18, 
                                    family = "Century Gothic", 
                                    color = "black"),
         axis.title.x = element_text(size = 18, 
                                     family = "Century Gothic", 
                                     color = "black",
                                     margin = margin(15, 0, 0, 0)),
         axis.text.y = element_text(size=18, 
                                    family = "Century Gothic", 
                                    color = "black"),
         axis.title.y = element_text(size=18, 
                                     family = "Century Gothic", 
                                     color = "black",
                                     margin = margin(0, 15, 0, 0)),
        axis.line.x = element_line(size=1.2),
        legend.position = "none")+
  scale_x_discrete(labels=rev(feat.label))+
  xlab("Features")+
  ylab("Average relative contribution to minimization of the objective function")

#predict the test data
temp.predict=predict(xgb.mod, dtest)
temp.predict=as.numeric(temp.predict > 0.4)
confusionMatrix(as.factor(temp.predict), as.factor(testing.df$collision_id), positive = "1")