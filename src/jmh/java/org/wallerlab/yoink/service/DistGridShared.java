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

public class DistGridShared implements ICalculateGridDistance
{
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
       cuModuleLoad(module, "DistGridShared.ptx");
       // Obtain a function pointer to the "add" function.
       CUfunction function = new CUfunction();
       cuModuleGetFunction(function, module, "distGridshared");
       
       
       int rows = 3;
       int elements = in1.length;
       int lengthAtom = in1.length;
       int lengthGrid = in2.length;  //Divided by GridPoints
       int Distances = lengthAtom*lengthGrid/9;// Distances = molecules * Atoms per Molecule * GridPoints
        
       // Allocate the device input data, and copy the
       // host input data to the device
       CUdeviceptr d_in1 = new CUdeviceptr();
       cuMemAlloc(d_in1, lengthAtom * Sizeof.FLOAT);
       cuMemcpyHtoD(d_in1, Pointer.to(in1),lengthAtom * Sizeof.FLOAT);
       
       CUdeviceptr d_in2 = new CUdeviceptr();
       cuMemAlloc(d_in2, lengthGrid  * Sizeof.FLOAT);
       cuMemcpyHtoD(d_in2, Pointer.to(in2),lengthGrid * Sizeof.FLOAT);
       
       CUdeviceptr d_out = new CUdeviceptr();
       cuMemAlloc(d_out, Distances * Sizeof.FLOAT);
       
       // Set up the kernel parameters: A pointer to an array
       // of pointers which point to the actual values.
       Pointer kernelParameters = Pointer.to(Pointer.to(d_in1),Pointer.to(d_in2),Pointer.to(d_out),
       Pointer.to(new int[]{lengthAtom}), Pointer.to(new int[]{lengthGrid}),Pointer.to(new int[]{Distances}) );
       
       // Call the kernel function.
       int blockSizeX = 960;
       int blockSizeY = 1;
       int gridSizeX = (elements/960)+1;
       int gridSizeY = 1;
       cuLaunchKernel(function,
           gridSizeX,  gridSizeY , 1,               // Grid dimension
           blockSizeX, blockSizeY, 1,      // Block dimension
           0, null,                        // Shared memory size and stream
           kernelParameters, null          // Kernel- and extra parameters
       );
       
       
       // Allocate host output memory and copy the device output
       // to the host.
       float out [] = new float [Distances];
       cuMemcpyDtoH(Pointer.to(out), d_out, Distances * Sizeof.FLOAT);
       
       cuMemFree(d_in1);
       cuMemFree(d_in2);
       cuMemFree(d_out);
       return out;
		
	}
	public float [] makingPoints (){
		float[] a = new float [6000];
		
		 for (int i = 0; i<6000;i++)
		  {
			Random random = new Random();
			a[i] = (float) (random.nextDouble()*10);
		  }
		return a;
	}

}

