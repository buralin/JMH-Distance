package org.wallerlab.yoink.service;

import java.util.List;

import org.wallerlab.yoink.domain.Atom;
import org.wallerlab.yoink.domain.Molecule;
import org.wallerlab.yoink.domain.Point;

public class DistanceCalculator implements IDistance
{
	
	public float[] calculateDistance(List<Molecule> molecules,Point com ) 
	{
		//number of atoms is currently set to 10 in the domain model. Multiply by three for three coords (x,y,z)
		float[] coordVector = new float [molecules.size()*10*3];	
		
		int atomIndex =0;
		for(int i = 0; i<molecules.size();i++){
			//System.out.println("MOLECULE SIZE " + molecules.size());
		    for(atomIndex=0; atomIndex < molecules.get(i).getAtoms().size();atomIndex ++){
		    	coordVector[atomIndex] = (float) molecules.get(i).getAtoms().get(atomIndex).getX();
				coordVector[atomIndex+(coordVector.length)/3] = (float)molecules.get(i).getAtoms().get(atomIndex).getY();
				coordVector[atomIndex+2*((coordVector.length)/3)] = (float) molecules.get(i).getAtoms().get(atomIndex).getZ();
		    }
		}
		//System.out.println("COORD VECTOR !!!!! *************** " +Arrays.toString(coordVector));
		
		
		float[] comVector = {(float) com.getX(),(float) com.getY(),(float) com.getZ()};
		
		
		return javadist(coordVector, comVector );
	}
	
	public float [] javadist (float [] in1, float [] in2)
	{
		int spalten = in1.length/3;
		float [] out = new float [spalten];
		for (int j=0; j < spalten; j++)
		{
				out [j] = (float) Math.sqrt((in1[j]-in2[0])*(in1[j+spalten]-in2[0])+
						(in1[j+spalten]-in2[1])*(in1[j+spalten]-in2[1])+
						(in1[j+2*spalten]-in2[2])*(in1[j+2*spalten]-in2[2]));
		}
		return out;
 }

}
