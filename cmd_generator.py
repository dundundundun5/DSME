CUDA = "1" # GPU number 0~3
MOE = "1" # set 1 use DSME 
H = 128 # MTSDNet-hidden
S = "tsg" # tsg-MTSDNet structure
D = 4 # MTSDNet- dim
REPEAT = 1 # repeat times
ONLY_INFO = 0 # set 1 to get model info without training
V = "../results" # save path
models = ['tccsnet','gruinc','elk','deepcoral','rsc','resnet','mtsdnet','ddnn','tripleattention','dualattention','gile','danncnn', 'deepconvlstm', 'transformer']
models = ['gruinc']
for BASELINE in models:
    # (dataset_name, sliding_window_length, batch_size, num_domains)
    datasets = [ ("dsads","32", "128", 8),("pamap2","128", "128", 8), ("opportunity", "64", "128", 4), ("uci_har","128","128", 4)]
    for i, dataset in enumerate(datasets):
        dataset_dict = eval(open("./data/dataset_dict.json", "r+").read())
        params =   {"dataset_name":"" ,  "model_name":"caonima" ,  "sliding_window_length": "" ,  "batch_size":"" ,  "epochs":"20" ,  "cuda_device":CUDA ,  "use_moe":MOE, "repeat": REPEAT, "end2end":"1", 'alpha':"0.5", "beta":"0.5", 'hidden':H, "structure_str":S, "save_path":V, "dim":D, "out_channels":64, "use_bagging":B, "only_info": ONLY_INFO} 
        params['model_name'] = BASELINE
        params["dataset_name"],  params['sliding_window_length'] , params['batch_size'], num_volunteers = dataset
        

        def split_generator(num_volunteers, use_moe):
            
            volunteer_splits = []
        # 1,2,3\|4
            use_moe = int(use_moe)
            if use_moe == 0:
                for i in range(1, num_volunteers+1):
                    
                    temp = f""
                    for j in range(1, num_volunteers+1):
                        
                        if j == i:
                            continue
                        temp += f"{j},"
                    temp = temp[:-1]
                    temp += f"\\|{i}"
                    volunteer_splits.append(temp)
            if use_moe == 1:
                for i in range(1, num_volunteers+1):
                    
                    temp = f""
                    for j in range(1, num_volunteers+1):
                        
                        if j == i:
                            continue
                        temp += f"{j}\\|"
                    temp = temp[:-2]
                    temp += f"\\|{i}"
                    volunteer_splits.append(temp)
            return volunteer_splits

        results = []
        volunteer_splits = split_generator(num_volunteers, params["use_moe"])
        for volunteer_split in volunteer_splits:
            result = "python main.py "
            for key,value in params.items():
                result += f"--{key} {value} "
            result += f"--volunteer_split {volunteer_split} "
            
            result += f"--n_domains {num_volunteers} "
            results.append(result)
            
            

        with open(f"./scripts/cuda{params['cuda_device']}.sh", mode="a+", encoding="utf-8") as f:
            
            print(f"{params['dataset_name']}_cuda{params['cuda_device']}")    
            for r in results:
                f.write(r + "\n\n")
                