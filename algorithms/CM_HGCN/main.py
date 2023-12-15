import time
from HGCN import *



def main():
    model = HGCN("Tmall", 0.001, 300, 1)
    model.fit()
    print("ab")
    start = time.time()
    best_result_k10 = [0, 0]
    best_result_k20 = [0, 0]
    best_result_k30 = [0, 0]
    best_result_k40 = [0, 0]
    best_result_k50 = [0, 0]
    
    
    best_epoch_k10 = [0, 0]
    best_epoch_k20 = [0, 0]
    bad_counter_k20 = bad_counter_k10 = 0
     
   
    hit_k10, mrr_k10, hit_k20, mrr_k20, hit_k30, mrr_k30, hit_k40, mrr_k40, hit_k50, mrr_k50 = model.predict_next()
    
    flag_k10 = 0
    if hit_k10 >= best_result_k10[0]:
        best_result_k10[0] = hit_k10
    
        flag_k10 = 1
    if mrr_k10 >= best_result_k10[1]:
        best_result_k10[1] = mrr_k10
        
        flag_k10 = 1   
    
    
    print("\n")
    print('Best @10 Result:')
    print('\tRecall@10:\t%.4f\tMMR@10:\t%.4f' % (
        best_result_k10[0], best_result_k10[1]))
    bad_counter_k10 += 1 - flag_k10
    
    flag_k20 = 0
    if hit_k20 >= best_result_k20[0]:
        best_result_k20[0] = hit_k20
        
        flag_k20 = 1
    if mrr_k20 >= best_result_k20[1]:
        best_result_k20[1] = mrr_k20

        flag_k20 = 1
    print('Best @20 Result:')
    print('\tRecall@20:\t%.4f\tMMR@20:\t%.4f' % (
        best_result_k20[0], best_result_k20[1]))
    bad_counter_k20 += 1 - flag_k20     
    
    
    if hit_k30 >= best_result_k30[0]:
        best_result_k30[0] = hit_k30
    if mrr_k30 >= best_result_k30[1]:
        best_result_k30[1] = mrr_k30
    print('Best @30 Result:')
    print('\tRecall@30:\t%.4f\tMMR@30:\t%.4f' % (
        best_result_k30[0], best_result_k30[1]))
    
    if hit_k40 >= best_result_k40[0]:
        best_result_k40[0] = hit_k40
    if mrr_k40 >= best_result_k40[1]:
        best_result_k40[1] = mrr_k40
    print('Best @40 Result:')
    print('\tRecall@40:\t%.4f\tMMR@40:\t%.4f' % (
        best_result_k40[0], best_result_k40[1]))
    
    
    if hit_k50 >= best_result_k50[0]:
        best_result_k50[0] = hit_k50
    if mrr_k50 >= best_result_k50[1]:
        best_result_k50[1] = mrr_k50
    print('Best @50 Result:')
    print('\tRecall@50:\t%.4f\tMMR@50:\t%.4f' % (
        best_result_k50[0], best_result_k50[1]))



        
        
        
        
    print('-------------------------------------------------------')
    end = time.time()
    print("Run time: %f s" % (end - start))


if __name__ == '__main__':
    main()
