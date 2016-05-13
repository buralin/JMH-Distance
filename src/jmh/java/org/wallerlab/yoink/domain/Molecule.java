package org.wallerlab.yoink.domain;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class Molecule implements MoleculeInterface {
	
	private final List<Atom> atoms = new ArrayList();
	
	private int index;
	
	
	public Molecule(int i){
		Random random = new Random();
		for(int j=0; j < 10; j++) atoms.add(new Atom());
		this.index = i;
	}


	public List<Atom> getAtoms() {
		return atoms;
	}
	
	
}
