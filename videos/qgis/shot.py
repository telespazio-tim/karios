import os
WORK = os.environ.get("VIDEO_WORK", "work") + "/"  # scratch dir: run outputs, crops, intermediates
from qgis.core import *
from qgis.gui import *
from qgis.utils import iface
from qgis.PyQt.QtCore import QTimer
from qgis.PyQt.QtWidgets import QApplication
from qgis.PyQt.QtCore import Qt
D=WORK+"demo/2/S2A_T30SYJ_20241105T105231_B04_clip_S2C_T30SYJ_20241105T105251_B04_work/"; OUT=WORK+"qgis/"
def go():
    mw=iface.mainWindow(); mw.showMaximized()
    for b in mw.findChildren(QgsMessageBar): b.clearWidgets()
    kp=QgsVectorLayer(D+"kp_delta.json","kp_delta","ogr"); ras=QgsRasterLayer("/home/gautier/projects/isl/issues/karios-3.0.0/2025-KARIOS_DEMO/2/WD/S2A_T30SYJ_20241105T105231_B04_clip.tif","S2A_B04 (monitored)")
    ras.setContrastEnhancement(QgsContrastEnhancement.StretchToMinimumMaximum,QgsRasterMinMaxOrigin.MinMax) if False else None
    QgsProject.instance().addMapLayers([kp,ras])
    ras.renderer().setContrastEnhancement(QgsContrastEnhancement(ras.dataProvider().dataType(1))); ce=ras.renderer().contrastEnhancement(); ce.setMinimumValue(1300); ce.setMaximumValue(5000); ce.setContrastEnhancementAlgorithm(QgsContrastEnhancement.StretchToMinimumMaximum)
    kp.renderer().symbol().setColor(QColor(230,40,40)); kp.renderer().symbol().setSize(2.2); kp.triggerRepaint()
    c=iface.mapCanvas(); feats=list(kp.getFeatures()); f=feats[len(feats)//3]
    pt=f.geometry().asPoint(); iface.setActiveLayer(kp)
    def full(): c.setExtent(ras.extent()); c.refresh()
    def zoom():
        r=QgsRectangle(pt.x()-900,pt.y()-500,pt.x()+900,pt.y()+500); c.setExtent(r); c.refresh()
    def ident():
        kp.selectByIds([f.id()]); c.refresh()
        form=QgsAttributeForm(kp,f,QgsAttributeEditorContext()); form.setMode(QgsAttributeEditorContext.Mode.SingleEditMode) if False else None
        form.resize(620,420); form.show(); QApplication.processEvents()
        form.grab().save(OUT+"form.png")
        QTimer.singleShot(1500,lambda: mw.grab().save(OUT+"q3.png"))
    def dump():
      try:
        _dump()
      except Exception:
        import traceback; print('TB',traceback.format_exc(),flush=True)
    def _dump():
        iface.actionIdentify().trigger()
        from qgis.PyQt.QtWidgets import QDockWidget
        print('DOCKS',[(d.objectName(),d.isVisible()) for d in mw.findChildren(QDockWidget)],flush=True)
        for d in mw.findChildren(QDockWidget):
            if 'dentify' in d.objectName() or 'dentify' in d.windowTitle(): d.setFloating(False); mw.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea,d); d.show(); d.setMinimumWidth(560); iface.messageBar().clearWidgets(); print('shown',d.objectName(),flush=True)
    pass
    def snap(n): mw.grab().save(OUT+n+".png")
    QTimer.singleShot(1500,full); QTimer.singleShot(5000,lambda:snap("q1"))
    QTimer.singleShot(6000,zoom); QTimer.singleShot(9500,lambda:snap("q2"))
    QTimer.singleShot(10500,zoom); QTimer.singleShot(13000,ident); 
    QTimer.singleShot(17000,QApplication.quit)
from qgis.PyQt.QtGui import QColor
QTimer.singleShot(8000,go)
