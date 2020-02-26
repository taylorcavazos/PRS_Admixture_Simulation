library(metafor)
library(data.table)

args = commandArgs(trailingOnly=TRUE)

se.from.p = function(ES, p) {
      z = qnorm(p)
      ES = log(ES)
      SE = abs(ES/z)
      return(SE)
}

rma.to.df = function(OR1,OR2,SE1,SE2) {
      vec = log(c(OR1,OR2))
      vec[is.na(vec)] = 0
      vec[is.infinite(vec)] = 0
      errors = c(SE1,SE2)
      errors[is.infinite(errors)] = NA
      errors[errors==0] = NA
      if (all(vec==0) | all(is.na(errors))) {
        return(c(1,1,NA,NA))
      }
      else {
        result = rma(vec,errors,method="FE")
        return(c(exp(result$beta),result$pval,result$QE,result$QEp))
      }
}

ceu_sum = fread(args[1])
yri_sum = fread(args[2])

ceu_sum$se = mapply(se.from.p, ceu_sum$OR, ceu_sum$`p-value`)
yri_sum$se = mapply(se.from.p, yri_sum$OR, yri_sum$`p-value`)

yri_sum_missing = merge(yri_sum,ceu_sum[,"var_id"],by="var_id",all.y = T,all.x = T)
ceu_sum_missing = merge(ceu_sum,yri_sum[,"var_id"],by="var_id",all.y = T,all.x = T)

yri_sum_missing[yri_sum_missing$`p-value`==1,"se"] = NA
ceu_sum_missing[ceu_sum_missing$`p-value`==1,"se"] = NA

meta_sum_stat = mapply(rma.to.df, ceu_sum_missing$OR,yri_sum_missing$OR,
       ceu_sum_missing$se,yri_sum_missing$se)
meta_sum_stat = t(as.data.frame(meta_sum_stat))
colnames(meta_sum_stat) = c("OR","p-value","Q","Q_p")
meta_sum_stat = cbind(var_id=ceu_sum_missing$var_id,meta_sum_stat)
write.table(file = args[3],x=meta_sum_stat,quote = F,sep = "\t",row.names = F)

