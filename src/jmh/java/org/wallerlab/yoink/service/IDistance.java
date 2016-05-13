package org.wallerlab.yoink.service;

import java.util.List;

import org.wallerlab.yoink.domain.Molecule;
import org.wallerlab.yoink.domain.Point;

public interface IDistance {

	 float[] calculateDistance(List<Molecule> molecules, Point com);

}