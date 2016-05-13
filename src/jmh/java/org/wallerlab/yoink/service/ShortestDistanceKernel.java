package org.wallerlab.yoink.service;

import static jcuda.driver.JCudaDriver.cuCtxCreate;
import static jcuda.driver.JCudaDriver.cuDeviceGet;
import static jcuda.driver.JCudaDriver.cuInit;
import static jcuda.driver.JCudaDriver.cuLaunchKernel;
import static jcuda.driver.JCudaDriver.cuMemAlloc;
import static jcuda.driver.JCudaDriver.cuMemFree;
import static jcuda.driver.JCudaDriver.cuMemcpyDtoH;
import static jcuda.driver.JCudaDriver.cuMemcpyHtoD;
import static jcuda.driver.JCudaDriver.cuModuleGetFunction;
import static jcuda.driver.JCudaDriver.cuModuleLoad;

import java.util.List;
import java.util.Random;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.driver.JCudaDriver;

import org.wallerlab.yoink.domain.GridPoint;
import org.wallerlab.yoink.domain.Molecule;
import org.wallerlab.yoink.domain.Point;

public class ShortestDistanceKernel implements ICalculateGridDistance {
	
	
	
	public float[] calculateDistance(List<Molecule> molecules, GridPoint grid) 
	{
		
		
	//number of atoms is currently set to 10 in the domain model. Multiply by three for three coords (x,y,z)
	float[] coordVector = new float [molecules.size()*10*3];
	float [] points = makingPoints();
 	int atomIndex =0;
	 for(int i = 0; i<molecules.size();i++)
	 {
	     for(atomIndex=0; atomIndex < molecules.get(i).getAtoms().size();atomIndex ++)
	     {
		    coordVector[atomIndex] = (float) molecules.get(i).getAtoms().get(atomIndex).getX();
			coordVector[atomIndex+(coordVector.length)/3] = (float)molecules.get(i).getAtoms().get(atomIndex).getY();
			coordVector[atomIndex+2*((coordVector.length)/3)] = (float) molecules.get(i).getAtoms().get(atomIndex).getZ();
		  }
	   }
			return distanceKernel(coordVector,points);
	}
	private float[] distanceKernel(float [] in1,float in2[]) { // 

	   // Enable exceptions and omit all subsequent error checks
       JCudaDriver.setExceptionsEnabled(true);
       
       // Initialize the driver and create a context for the first device.
       cuInit(0);
       CUdevice device = new CUdevice();
       cuDeviceGet(device, 0);
       CUcontext context = new CUcontext();
       cuCtxCreate(context, 0, device);
       
       
       // Load the ptx file.
       CUmodule module = new CUmodule();
       cuModuleLoad(module, "DistGrid.ptx");
       // Obtain a function pointer to the "add" function.
       CUfunction function = new CUfunction();
       cuModuleGetFunction(function, module, "distGrid");
       
       
       int rows = 3;
       int elements = in1.length;
       int columns1 = in1.length/3 ;
       int columns2 = in2.length/3;  //Divided by GridPoints
       int distances = columns1*columns2;// Distances = molecules * Atoms per Molecule * GridPoints
        
       // Allocate the device input data, and copy the
       // host input data to the device
       CUdeviceptr d_in1 = new CUdeviceptr();
       cuMemAlloc(d_in1, columns1 *rows* Sizeof.FLOAT);
       cuMemcpyHtoD(d_in1, Pointer.to(in1),columns1*rows * Sizeof.FLOAT);
       
       CUdeviceptr d_in2 = new CUdeviceptr();
       cuMemAlloc(d_in2, columns2 * rows * Sizeof.FLOAT);
       cuMemcpyHtoD(d_in2, Pointer.to(in1),columns2*rows * Sizeof.FLOAT);
       
       CUdeviceptr d_out = new CUdeviceptr();
       cuMemAlloc(d_out, distances * Sizeof.FLOAT);
       
       // Set up the kernel parameters: A pointer to an array
       // of pointers which point to the actual values.
       Pointer kernelParameters = Pointer.to(Pointer.to(d_in1),Pointer.to(d_in2),Pointer.to(d_out),
       Pointer.to(new int[]{columns1}), Pointer.to(new int[]{columns2}));
       
       // Call the kernel function.
       int blockSizeX = 1024;
       int blockSizeY = 1;
       int gridSizeX = (elements +blockSizeX -1)/blockSizeX;
       int gridSizeY = 1;
       cuLaunchKernel(function,
           gridSizeX,  gridSizeY , 1,               // Grid dimension
           blockSizeX, blockSizeY, 1,      // Block dimension
           0, null,                        // Shared memory size and stream
           kernelParameters, null          // Kernel- and extra parameters
       );
       
       
       // Allocate host output memory and copy the device output
       // to the host.
       float out [] = new float [distances];
       cuMemcpyDtoH(Pointer.to(out), d_out, distances * Sizeof.FLOAT);
       
       cuMemFree(d_in1);
       cuMemFree(d_in2);
       cuMemFree(d_out);
       return out;
		
	}
	public float [] makingPoints (){
		float[] a = new float [3000];
		
		 for (int i = 0; i<3000;i++)
		  {
			Random random = new Random();
			a[i] = (float) (random.nextDouble()*10);
		  }
		return a;
	}

}
