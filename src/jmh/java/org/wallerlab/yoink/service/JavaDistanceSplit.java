package org.wallerlab.yoink.service;

import java.util.List;

import org.wallerlab.yoink.domain.Molecule;
import org.wallerlab.yoink.domain.Point;

public class JavaDistanceSplit implements IDistance
	{
	    public float[] calculateDistance (List<Molecule> molecules,Point com )
		{
		double [] comVector = {com.getX(),com.getY(), com.getZ()};
		int atomIndex =0;
		double dist = 0;
		float [] distarray = new float [molecules.get(0).getAtoms().size()];
		for(int i = 0; i<molecules.size();i++)
		{
		    for(atomIndex=0; atomIndex < molecules.get(i).getAtoms().size();atomIndex ++)
		    {
		    	double [] in1 = new double [3];
		    	in1[0] = molecules.get(i).getAtoms().get(atomIndex).getX();
		    	in1[1] = molecules.get(i).getAtoms().get(atomIndex).getY();
		    	in1[2] = molecules.get(i).getAtoms().get(atomIndex).getZ();
		        dist = javadist (in1, comVector);
		        distarray [atomIndex] = (float)dist;
		    } 
		}
		return distarray;
	    }
	    
		public double javadist (double [] in1, double [] in2)
		{
		double out = 0;
		
		out =  Math.sqrt((in1[0]-in2[0])*(in1[0]-in2[0])+
							(in1[1]-in2[1])*(in1[1]-in2[1])+
							(in1[2]-in2[2])*(in1[2]-in2[2]));
		return out;
	   }
	
}

