import sys 
import argparse
import multiprocessing


sys.path.append('.')

from detect import run 

def run_camera (source_id, weights,name_suffix):
    run(
        weights=weights,
        source=source_id,
        imgsz=(720,720),
        conf_thres=0.30,
        iou_thres=0.45,
        device='0',
        project='runs\multi',
        name=f'cam{name_suffix}',
        exist_ok=True
    )

if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights',type=str,default='best.pt', help='direktori ke bobot')
    parser.add_argument('--sources',nargs='+',required=True,help='senarai kamera atau tautan')

    args = parser.parse_args()

    processes = []

    for i , cam in enumerate(args.sources):
        source = int(cam) if cam.isdigit() else cam
        p = multiprocessing.Process(target=run_camera,args=(source,args.weights,i))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


