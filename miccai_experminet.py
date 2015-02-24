# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 14:36:56 2015

@author: nusrat
"""

import numpy as np

from dipy.tracking.distances import bundles_distances_mam
from dipy.tracking.metrics import length
from dipy.align.streamlinear import StreamlineLinearRegistration
from dipy.tracking.metrics import downsample
from nibabel import trackvis
from fastPFP import fastPFP_faster, greedy_assignment
from joblib import cpu_count, Parallel, delayed

def bundles_distances_mam_smarter(A, B=None):
    """Smarter of bundles_distances_mam that avoids computing
    distances twice.
    """
    lenA = len(A)
    if B is None:
        dm = np.empty((lenA, lenA), dtype=np.float32)
        dm[np.diag_indices(lenA)] = 0.0
        for i, s in enumerate(A[:-1]):
            dm[i, i+1:] = bundles_distances_mam([s], A[i+1:])
            dm[i+1:, i] = dm[i, i+1:]

    else:
        lenB = len(B)
        dm = np.empty((lenA, lenB), dtype=np.float32)
        for i, s in enumerate(A):
            dm[i, :] = bundles_distances_mam([s], B)

    return dm


def chunker(seq, size):
    return (seq[pos:pos + size] for pos in xrange(0, len(seq), size))


def bundles_distances_mam_smarter_faster(A, B=None, n_jobs=-1, chunk_size=100):
    """Parallel version of bundles_distances_mam that also avoids
    computing distances twice.
    """
    lenA = len(A)
    chunks = chunker(A, chunk_size)
    if B is None:
        dm = np.empty((lenA, lenA), dtype=np.float32)
        dm[np.diag_indices(lenA)] = 0.0
        results = Parallel(n_jobs=-1)(delayed(bundles_distances_mam)(ss, A[i*chunk_size+1:]) for i, ss in enumerate(chunks))
        # Fill triu
        for i, res in enumerate(results):
            dm[(i*chunk_size):((i+1)*chunk_size), (i*chunk_size+1):] = res
            
        # Copy triu to trid:
        rows, cols = np.triu_indices(lenA, 1)
        dm[cols, rows] = dm[rows, cols]

    else:
        dm = np.vstack(Parallel(n_jobs=n_jobs)(delayed(bundles_distances_mam)(ss, B) for ss in chunks))

    return dm
def voxel_measure(static_center, moving_center, show=False, vol_size=(256, 300, 256)):
    
    vol_A = np.zeros(vol_size)

    ci, cj, ck = vol_size[0] / 2, vol_size[1] / 2, vol_size[2] / 2

    spts = np.concatenate(static_center, axis=0)
    spts = np.round(spts).astype(np.int) + np.array([ci, cj, ck])

    mpts = np.concatenate(moving_center, axis=0)
    mpts = np.round(mpts).astype(np.int) + np.array([ci, cj, ck])

    for index in spts:
        i, j, k = index
        vol_A[i, j, k] = 1

    vol_B = np.zeros(vol_size)
    
    for index in mpts:
        i, j, k = index
        vol_B[i, j, k] = 1

    vol_and = np.logical_and(vol_A, vol_B)
    
    return   np.sum(vol_and),np.sum(vol_B)
    
    
    
def voxel_based_registration( source_tractography_streamlines, target_tractography_streamlines,subject_pair):
    
    intersection_voxel_list=[]   
    target_voxel_list=[]
    
 
  
    for i in range(len(source_tractography_streamlines)):
             voxel_and,voxel_target=voxel_measure( source_tractography_streamlines[ source+'_'+subject_tracts[str(subject_pair)][i]][0]
                                                  ,target_tractography_streamlines[ target+'_'+subject_tracts[str(subject_pair)][i]][0])
             intersection_voxel_list.append( voxel_and)
             target_voxel_list.append(voxel_target)
             
    total_intersection_voxel_list =np.sum(np.array(intersection_voxel_list))        
    total_target_voxel_list =np.sum(np.array(target_voxel_list))
   
    print "Number of voxel per tract"
    print intersection_voxel_list,target_voxel_list
    
    print "Number of voxel"
    print  total_intersection_voxel_list,total_target_voxel_list
    TDA_all_voxel_registration=np.divide( total_intersection_voxel_list,total_target_voxel_list)
    
    print "Modified-TDR-for-all"
    print TDA_all_voxel_registration
   
def streamline_based_registration( source_tractography_streamlines, target_tractography_streamlines,subject_pair):
    
    intersection_voxel_list=[]   
    target_voxel_list=[]
    n_points=20
    srr = StreamlineLinearRegistration()
    SAr = [downsample(i, n_points) for i in source_tractography]
    SBr = [downsample(i, n_points) for i in target_tractography]
    srm = srr.optimize(static=SBr, moving=SAr)
    transformed_tractography = srm.transform(source_tractography)
    
    print len(transformed_tractography )
    temp_index=0
    for i in range(len(source_tractography_streamlines)):
            
             voxel_and,voxel_target=voxel_measure( transformed_tractography[temp_index:temp_index+source_tractography_streamlines[ source+'_'+subject_tracts[str(subject_pair)][i]][1]]
                                                  ,target_tractography_streamlines[ target+'_'+subject_tracts[str(subject_pair)][i]][0])
                                                  
             temp_index=temp_index+ source_tractography_streamlines[ source+'_'+subject_tracts[str(subject_pair)][i]][1]        
                              
             intersection_voxel_list.append( voxel_and)
             target_voxel_list.append(voxel_target)
             
    total_intersection_voxel_list =np.sum(np.array(intersection_voxel_list))        
    total_target_voxel_list =np.sum(np.array(target_voxel_list))
    
    print "Number of voxel per tract"
    print intersection_voxel_list,target_voxel_list
    
    print "Number of voxel"
    print  total_intersection_voxel_list,total_target_voxel_list
    TDA_all_voxel_registration=np.divide( total_intersection_voxel_list,total_target_voxel_list)
    
    print "Modified-TDR-for-all"
    print TDA_all_voxel_registration
   
def mapping( source_tractography_streamlines, target_tractography_streamlines,subject_pair):
    
    intersection_voxel_list=[]   
    target_voxel_list=[]
    
    dmA = bundles_distances_mam_smarter_faster(source_tractography) 
    dmB = bundles_distances_mam_smarter_faster(target_tractography) 
   
    print("Computing similarity matrices.")
    #sigma = np.mean([np.median(dmA), np.median(dmB)])
    sigmaA = np.median(dmA)
    sigmaB = np.median(dmB)
    A = np.exp(- dmA / sigmaA)
    B = np.exp(- dmB / sigmaB)
    X = fastPFP_faster(B, A, alpha=0.9, threshold1=1.0e-4, threshold2=1.0e-4, max_iter1=200, max_iter2=200)
    
    P = greedy_assignment(X)
   
    mapping_idx=P.argmax(0)
    mapped_tractography=[target_tractography[i] for i in mapping_idx]
    out_name = "/home/nusrat/clu_com/mapping_result/"+subject_pair+'multiple_tract'+'.txt'
    np.savetxt(out_name,mapping_idx)
    #out_name = "/home/nusrat/clu_com/mapping_result/"+source+target+".txt"
    #np.savetxt(out_name,mapping)
    
    print len(mapped_tractography )
    temp_index=0
    for i in range(len(source_tractography_streamlines)):
            
             voxel_and,voxel_target=voxel_measure( mapped_tractography[temp_index:temp_index+source_tractography_streamlines[ source+'_'+subject_tracts[str(subject_pair)][i]][1]]
                                                  ,target_tractography_streamlines[ target+'_'+subject_tracts[str(subject_pair)][i]][0])
                                                  
             temp_index=temp_index+ source_tractography_streamlines[ source+'_'+subject_tracts[str(subject_pair)][i]][1]        
                              
             intersection_voxel_list.append( voxel_and)
             target_voxel_list.append(voxel_target)
             
    total_intersection_voxel_list =np.sum(np.array(intersection_voxel_list))        
    total_target_voxel_list =np.sum(np.array(target_voxel_list))
    
    print "Number of voxel per tract"
    print intersection_voxel_list,target_voxel_list
    
    print "Total Number of voxel"
    print  total_intersection_voxel_list,total_target_voxel_list
    TDA_all_voxel_registration=np.divide( total_intersection_voxel_list,total_target_voxel_list)
    
    
    print "Modified-TDR-for-all"
    print TDA_all_voxel_registration
   
   
def load_tractography(subject_id,subject_pair):
    
    tract_streamline={}
    Total_streamline=0
    
    for tract in subject_tracts[str(subject_pair)]:
     
      filename_tractography = '/home/nusrat//targetDirectory/MICCAI2015_DTI_EUDX/'+str(subject_id)+'_1M_wmql/wmql_'+str(tract)+'.trk'
    
      tractography, header = trackvis.read(filename_tractography, points_space='voxel') 
      tractography = [streamline[0] for streamline in tractography]
      
      tractography = [streamline for streamline in tractography if length(streamline)>= 15] 
      dict_element=(tractography,len(tractography))
      tract_streamline[str(subject_id)+'_'+str(tract)]=dict_element
      
      
      Total_streamline=len( tractography)+ Total_streamline
    
   
    return tract_streamline,Total_streamline


     
if __name__ == '__main__':
    
   global source,target,source_tratography,target_tractography,subject_pair 

   subject_id = ["100307", "124422", "161731", "199655", "201111"]
   
   subject_tracts = {'100307-124422':["cg.left", "cg.right", "ifof.left", "ifof.right", "or.right", "uf.left", "uf.right"],
                     '100307-161731':["cg.left", "cg.right", "ifof.left", "mdlf.right", "mdlf.left", "uf.left"],
                     '100307-199655':["cg.left", "cg.right", "mdlf.left", "or.right", "uf.left", "uf.right"],
                     '100307-201111':["cg.left", "cg.right", "ifof.left", "ifof.right", "or.right", "uf.left", "uf.right"],

                     '124422-161731':["cst.right","cg.left", "cg.right","af.left","af.right","ifof.left", "uf.left", "uf.right"],  
                     '124422-199655':["cst.right", "cg.left", "cg.right","af.left", "ifof.left","ifof.right", "mdlf.right","or.right","ilf.left"],
                     '124422-201111':["cg.left", "cg.right", "af.left", "or.right", "uf.right"],

                     '161731-199655':["cg.left", "cg.right", "af.left", "mdlf.left", "uf.left"],  
                     '161731-201111':["cst.right", "cg.left", "cg.right", "af.left",  "ifof.left", "mdlf.right", "or.right"], 
   
	               '199655-201111':["cg.left", "af.left","cg.right", "mdlf.right",  "or.right", "uf.left", "uf.right"] 
		      }

   First_subject_id=subject_id[3]
   Second_subject_id=subject_id[4]
   subject_pair=str(First_subject_id)+'-'+str(Second_subject_id)
   
   
   tractograpy_1_streamlines,total_no_of_streamline_tractograpy_1=load_tractography(First_subject_id,subject_pair)
   tractograpy_2_streamlines,total_no_of_streamline_tractograpy_2=load_tractography(Second_subject_id,subject_pair)
   
 
   #tractograpy_1=np.concatenate(np.array([ tractograpy_1_streamlines[i][0] for i in tractograpy_1_streamlines.keys()]))    
   #tractograpy_2=np.concatenate(np.array([ tractograpy_2_streamlines[i][0] for i in tractograpy_2_streamlines.keys()]))  
   
   
   if total_no_of_streamline_tractograpy_1>total_no_of_streamline_tractograpy_2:
                    source_tractography_streamlines= tractograpy_2_streamlines
                    target_tractography_streamlines= tractograpy_1_streamlines
                    
   if total_no_of_streamline_tractograpy_1<total_no_of_streamline_tractograpy_2:
                    source_tractography_streamlines= tractograpy_1_streamlines
                    target_tractography_streamlines= tractograpy_2_streamlines
                    

   source=source_tractography_streamlines.keys()[0].split("_")[0]
   target=target_tractography_streamlines.keys()[0].split("_")[0]
   print ("source=%s" %source)
   print ("target=%s" %target)
   
   source_tractography=np.concatenate(np.array([ source_tractography_streamlines[ source+'_'+subject_tracts[str(subject_pair)][i]][0] for i in range(len(source_tractography_streamlines))]))
   target_tractography=np.concatenate(np.array([ target_tractography_streamlines[ target+'_'+subject_tracts[str(subject_pair)][i]][0] for i in range(len(source_tractography_streamlines))]))
   print len(source_tractography),  len( target_tractography)  
 
   print "Voxel-Based Registration"  
   voxel_based_registration( source_tractography_streamlines, target_tractography_streamlines,subject_pair)   
     
   print "Streamline-Based Registration"    
   streamline_based_registration( source_tractography_streamlines, target_tractography_streamlines,subject_pair)   
   
   print "Mapping"    
   mapping( source_tractography_streamlines, target_tractography_streamlines,subject_pair)                    
   # a=[source_tractography_streamlines[ source+'_'+subject_tracts[str(subject_pair)][i]][1]for i in range(len(source_tractography_streamlines))]
   #  [527, 565, 179, 217, 57, 118, 165]
             
   
   
