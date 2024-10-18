import numpy as np
import imagej
import scyjava as sj
#ij = imagej.init(mode='interactive')
ij = imagej.init(mode='interactive')
def blob_dector(bg_filename, sg_filename):
    try:
        image = ij.io().open(bg_filename)
        image_base = ij.io().open(bg_filename)
        imp = ij.py.to_imageplus(image)
        imp_base = ij.py.to_imageplus(image_base)
        sg_imp = ij.py.to_imageplus(ij.io().open(sg_filename))
        Prefs = sj.jimport('ij.Prefs')
        Prefs.blackBackground = True
        ij.IJ.setAutoThreshold(imp, "Otsu dark")
        ImagePlus = sj.jimport("ij.ImagePlus")
        mask = ImagePlus("cells-mask", imp.createThresholdMask())
        ij.IJ.run(imp, "Close", "")
        ij.IJ.run(mask, "Watershed", "")
        ij.py.show(mask)
        ij.IJ.run("Set Measurements...", "area min centroid redirect=None decimal=3")
        ij.IJ.run(mask, "Analyze Particles...", "size=10-180 circularity=0.60-1.00 exclude clear add composite");
        rm = ij.RoiManager.getInstance()
        array = rm.getRoisAsArray()
        ij.IJ.run(mask, "Close", "")
        Table = sj.jimport('org.scijava.table.Table')
        rt = ij.ResultsTable.getResultsTable()
        for roi in array:
            imp_base.setRoi(roi)
            ij.IJ.run(imp_base,"Measure", "")
        sci_table = ij.convert().convert(rt, Table)
        df = ij.py.from_java(sci_table)
        ij.IJ.selectWindow("Results"); 
        ij.IJ.run("Close");
        rt.reset()
        for roi in array:
            sg_imp.setRoi(roi)
            ij.IJ.run(sg_imp,"Measure", "")
        sci_table = ij.convert().convert(rt, Table)
        sg_df = ij.py.from_java(sci_table)
        ij.IJ.selectWindow("Results"); 
        ij.IJ.run("Close");
        rm.close();
        return df, sg_df

    except Exception as e:
        print('error', e)
        return None, None
