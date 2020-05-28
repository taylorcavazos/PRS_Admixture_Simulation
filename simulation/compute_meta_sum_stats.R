# R functions to perform meta analysis from summary statistics
# in Africans and Europeans

# Install R packages
options(install.packages.compile.from.source = "always")
if (!require("metafor")) install.packages("metafor")
library(metafor)
library(data.table)

# Read in summary statistics
args = commandArgs(trailingOnly=TRUE)

# Convert p-value to standard error
se.from.p = function(ES, p) {
      z = qnorm(p)
      ES = log(ES)
      SE = abs(ES/z)
      return(SE)
}

# Computes the meta for each variant
rma.to.df = function(OR1,OR2,SE1,SE2) {
      # convert ORs to log ORs 
      vec = log(c(OR1,OR2))
      # replace infinite and missing values with 0 
      vec[is.na(vec)] = 0
      vec[is.infinite(vec)] = 0
      errors = c(SE1,SE2)
      # replace infinite and missing values with NA
      errors[is.infinite(errors)] = NA
      errors[errors==0] = NA
      # if variant is missing in both studies output OR and p-value of 1
      if (all(vec==0) | all(is.na(errors))) {
        return(c(1,1,NA,NA))
      }
      # if variant present in at least one study, perform FE meta
      # in the case of a single study the inverse variance weighted 
      # effect is the result
      else {
        result = rma(vec,errors,method="FE")
        return(c(exp(result$beta),result$pval,result$QE,result$QEp))
      }
}

# Read European summary statistics
ceu_sum = fread(args[1])
# Read African summary statistics
yri_sum = fread(args[2])

# Calculate standard errors
ceu_sum$se = mapply(se.from.p, ceu_sum$OR, ceu_sum$`p-value`)
yri_sum$se = mapply(se.from.p, yri_sum$OR, yri_sum$`p-value`)
# Merge summary statistics
yri_sum_missing = merge(yri_sum,ceu_sum[,"var_id"],by="var_id",all.y = T,all.x = T)
ceu_sum_missing = merge(ceu_sum,yri_sum[,"var_id"],by="var_id",all.y = T,all.x = T)
# Standard errors for p=1 are inf, replace with NA
yri_sum_missing[yri_sum_missing$`p-value`==1,"se"] = NA
ceu_sum_missing[ceu_sum_missing$`p-value`==1,"se"] = NA
# Perform meta and output results
meta_sum_stat = mapply(rma.to.df, ceu_sum_missing$OR,yri_sum_missing$OR,
       ceu_sum_missing$se,yri_sum_missing$se)
meta_sum_stat = t(as.data.frame(meta_sum_stat))
colnames(meta_sum_stat) = c("OR","p-value","Q","Q_p")
meta_sum_stat = cbind(var_id=ceu_sum_missing$var_id,meta_sum_stat)
write.table(file = args[3],x=meta_sum_stat,quote = F,sep = "\t",row.names = F)

