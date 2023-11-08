# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 14:51:09 2020
@ProjectTitle : Thermal Mixed-Hydrodynamic Lubrication Solver for simulation of a compression ring - cylinder liner contact in an IC Engine.
@author: Dieter Fauconnier
@version : 1.0
"""

import h5py
import copy
import numpy as np


class IOHDF5:

    def Convert2Dict(self,VarDict : dict):
        ObjectDictionary=copy.deepcopy(VarDict)
        
        #Level 1        
        NameList=list(ObjectDictionary.keys())
        for k1 in range(len(ObjectDictionary)):
            if hasattr(ObjectDictionary[NameList[k1]], "__dict__"): #check if object is a class
               ObjectDictionary[NameList[k1]] = self.GetMembers(ObjectDictionary[NameList[k1]])
               
               #Level 2
               NameList2=list(ObjectDictionary[NameList[k1]].keys())
               for k2 in range(len(ObjectDictionary[NameList[k1]])):
                   if hasattr(ObjectDictionary[NameList[k1]][NameList2[k2]], "__dict__"): #check if object is a class
                      ObjectDictionary[NameList[k1]][NameList2[k2]] = self.GetMembers(ObjectDictionary[NameList[k1]][NameList2[k2]]) 
                      
                      #Level 3
                      NameList3=list(ObjectDictionary[NameList[k1]][NameList2[k2]].keys())
                      for k3 in range(len(ObjectDictionary[NameList[k1]][NameList2[k2]])):
                          if hasattr(ObjectDictionary[NameList[k1]][NameList2[k2]][NameList3[k3]], "__dict__"): #check if object is a class
                             ObjectDictionary[NameList[k1]][NameList2[k2]][NameList3[k3]] = self.GetMembers(ObjectDictionary[NameList[k1]][NameList2[k2]][NameList3[k3]])
        return ObjectDictionary
      
    def GetMembers(self,Object):
        Object = vars(Object)
        return Object
    
    def OpenHDF5(self,FileName : str='Save',Mode : str='w'):
        f = h5py.File(FileName,Mode)
        return f

    def CloseHDF5(self,FileHandle):
        FileHandle.flush()
        FileHandle.close()
        
    
    def Write2HDF5(self,f,VarDict : dict):
        # Convert objects to nested dictionary
        PrimitiveDictionary=self.Convert2Dict(VarDict)
        
        # Start writing structure
        #Level 1
        NameList=list(PrimitiveDictionary.keys())
        for k1 in range(len(NameList)):
            arr=PrimitiveDictionary[NameList[k1]]
            if type(arr)==dict:
                group = f.create_group(NameList[k1])
                
                #Level 2
                NameList2=list(arr.keys())
                for k2 in range(len(NameList2)):
                    arr2=arr[NameList2[k2]]
                    if type(arr2)==dict:
                        group2 = group.create_group(NameList2[k2])
                        
                        #Level 3
                        NameList3=list(arr2.keys())
                        for k3 in range(len(NameList3)):
                            arr3=arr2[NameList3[k3]]
                            if type(arr3)==dict:
                                group3 = group2.create_group(NameList3[k3])
                                
                                #Level 4
                                NameList4=list(arr3.keys())
                                for k4 in range(len(NameList4)):
                                    arr4=arr3[NameList4[k4]]
                                    if type(arr4)==dict:
                                        group4 = group3.create_group(NameList4[k4])
                                    else:
                                        self.WriteDataSet(group3,arr4,NameList4[k4])
                            else:
                                self.WriteDataSet(group2,arr3,NameList3[k3])
                    else:
                        self.WriteDataSet(group,arr2,NameList2[k2])
            else:
                self.WriteDataSet(f,arr,NameList[k1])


    def WriteDataSet(self,f,arr,Name):
        if np.size(arr)>1:
            dset = f.create_dataset(Name, np.shape(arr), dtype=arr.dtype)
            dset[:]=arr
        elif type(arr)==str:
            dset=f.create_dataset(Name,(1,),'S10')
            dset[0] = np.string_(arr)
        else :
            dset=f.create_dataset(Name,(1,1),type(arr).__name__)
            dset[0]=np.array(arr)


    def ReadHDF5(self,f):
        
        DataDictionary={}
        
        NameList=list(f.keys())
        for k in range(len(NameList)):
            n1 = f.get(NameList[k])
            DataDictionary[NameList[k]]={}
            
            if not isinstance(n1, h5py.Dataset):
                NameList2=list(n1.keys())
                for k2 in range(len(NameList2)):
                    n2 = (n1.get(NameList2[k2]))
                    DataDictionary[NameList[k]][NameList2[k2]]={}
                    
                    if not isinstance(n2, h5py.Dataset):
                        NameList3=list(n2.keys())
                        for k3 in range(len(NameList3)):
                            n3 = (n2.get(NameList3[k3]))
                            DataDictionary[NameList[k]][NameList2[k2]][NameList3[k3]]={}
                            
                            if not isinstance(n3, h5py.Dataset):
                                NameList4=list(n3.keys())
                                for k4 in range(len(NameList4)):
                                    n4 = (n3.get(NameList4[k4]))
                                    DataDictionary[NameList[k]][NameList2[k2]][NameList3[k3]][NameList4[k4]]={}
                                    if isinstance(n4, h5py.Dataset):
                                        DataDictionary[NameList[k]][NameList2[k2]][NameList3[k3]][NameList4[k4]]=np.array(n4)
                                    else:
                                        print('Warning IO:: HDF5 Reading level is too deep!!')
                            else:
                                DataDictionary[NameList[k]][NameList2[k2]][NameList3[k3]]=np.array(n3)
                    
                    else:
                        DataDictionary[NameList[k]][NameList2[k2]]=np.array(n2)
            else:
                DataDictionary[NameList[k]]=np.array(n1)               
        
        return DataDictionary



    def SaveData(self,FileName,Data2File):
        ID=self.OpenHDF5(FileName,'w')
        try:
            self.Write2HDF5(ID,Data2File)
            print('IO:: '+FileName+': Data written successfully!!')
            self.CloseHDF5(ID)
        except:
            print('Warning IO:: '+FileName+': Data could not be written!!')
            self.CloseHDF5(ID)
            
            
    def ReadData(self,FileName):
        ID=self.OpenHDF5(FileName,'r')
        try:
            DataDictionary=self.ReadHDF5(ID)
            print('IO:: '+FileName+': Data read successfully!!')
            self.CloseHDF5(ID)
        except:
            print('Warning IO:: '+FileName+': Data could not be read!!')
            self.CloseHDF5(ID)
        return DataDictionary