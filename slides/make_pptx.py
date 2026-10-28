"""Convert the compiled Beamer PDF into a 16:9 PowerPoint deck (one full-slide image per page).

    cd slides && python make_pptx.py          # needs only pdftoppm (poppler); no Python packages

The .pptx is written directly as Office Open XML, so the LaTeX maths and figures look identical.
Edit the .tex, recompile the PDF, then rerun this script.
"""
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PDF = HERE / "lstd_presentation.pdf"
PPTX = HERE / "lstd_presentation.pptx"
DPI = 250
CX, CY = 12192000, 6858000  # 13.333 x 7.5 in (16:9) in EMU

NS = (
    'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
    'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
    'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"'
)
HDR = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
CT = "application/vnd.openxmlformats-officedocument.presentationml"
EMPTY_TREE = (
    '<p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>'
    '<p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/>'
    '<a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>'
)


def rels(*items: tuple[str, str, str]) -> str:
    body = "".join(f'<Relationship Id="{i}" Type="{t}" Target="{tgt}"/>' for i, t, tgt in items)
    return f'{HDR}<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">{body}</Relationships>'


def theme() -> str:
    accents = ["4F81BD", "C0504D", "9BBB59", "8064A2", "4BACC6", "F79646"]
    colors = (
        '<a:dk1><a:sysClr val="windowText" lastClr="000000"/></a:dk1>'
        '<a:lt1><a:sysClr val="window" lastClr="FFFFFF"/></a:lt1>'
        '<a:dk2><a:srgbClr val="1F497D"/></a:dk2><a:lt2><a:srgbClr val="EEECE1"/></a:lt2>'
        + "".join(f'<a:accent{i}><a:srgbClr val="{c}"/></a:accent{i}>' for i, c in enumerate(accents, 1))
        + '<a:hlink><a:srgbClr val="0000FF"/></a:hlink><a:folHlink><a:srgbClr val="800080"/></a:folHlink>'
    )
    font = '<a:latin typeface="Calibri"/><a:ea typeface=""/><a:cs typeface=""/>'
    fill = '<a:solidFill><a:schemeClr val="phClr"/></a:solidFill>'
    line = f'<a:ln w="9525">{fill}</a:ln>'
    effect = "<a:effectStyle><a:effectLst/></a:effectStyle>"
    return (
        f'{HDR}<a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" name="Office Theme">'
        f'<a:themeElements><a:clrScheme name="Office">{colors}</a:clrScheme>'
        f'<a:fontScheme name="Office"><a:majorFont>{font}</a:majorFont><a:minorFont>{font}</a:minorFont></a:fontScheme>'
        f'<a:fmtScheme name="Office"><a:fillStyleLst>{fill * 3}</a:fillStyleLst>'
        f'<a:lnStyleLst>{line * 3}</a:lnStyleLst>'
        f'<a:effectStyleLst>{effect * 3}</a:effectStyleLst>'
        f'<a:bgFillStyleLst>{fill * 3}</a:bgFillStyleLst></a:fmtScheme></a:themeElements>'
        '<a:objectDefaults/><a:extraClrSchemeLst/></a:theme>'
    )


def slide_xml(n: int) -> str:
    return (
        f'{HDR}<p:sld {NS}><p:cSld><p:spTree>{EMPTY_TREE}'
        f'<p:pic><p:nvPicPr><p:cNvPr id="2" name="Slide {n}"/>'
        '<p:cNvPicPr><a:picLocks noChangeAspect="1"/></p:cNvPicPr><p:nvPr/></p:nvPicPr>'
        '<p:blipFill><a:blip r:embed="rId2"/><a:stretch><a:fillRect/></a:stretch></p:blipFill>'
        f'<p:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{CX}" cy="{CY}"/></a:xfrm>'
        '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></p:spPr></p:pic>'
        '</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sld>'
    )


def build(pages: list[Path], out: Path) -> None:
    n = len(pages)
    overrides = [
        ("/ppt/presentation.xml", f"{CT}.presentation.main+xml"),
        ("/ppt/slideMasters/slideMaster1.xml", f"{CT}.slideMaster+xml"),
        ("/ppt/slideLayouts/slideLayout1.xml", f"{CT}.slideLayout+xml"),
        ("/ppt/theme/theme1.xml", "application/vnd.openxmlformats-officedocument.theme+xml"),
        ("/ppt/presProps.xml", f"{CT}.presProps+xml"),
        ("/ppt/viewProps.xml", f"{CT}.viewProps+xml"),
        ("/ppt/tableStyles.xml", f"{CT}.tableStyles+xml"),
        ("/docProps/core.xml", "application/vnd.openxmlformats-package.core-properties+xml"),
        ("/docProps/app.xml", "application/vnd.openxmlformats-officedocument.extended-properties+xml"),
    ] + [(f"/ppt/slides/slide{i}.xml", f"{CT}.slide+xml") for i in range(1, n + 1)]
    content_types = (
        f'{HDR}<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Default Extension="png" ContentType="image/png"/>'
        + "".join(f'<Override PartName="{p}" ContentType="{t}"/>' for p, t in overrides)
        + "</Types>"
    )
    slide_ids = "".join(f'<p:sldId id="{255 + i}" r:id="rId{i + 4}"/>' for i in range(1, n + 1))
    presentation = (
        f'{HDR}<p:presentation {NS} saveSubsetFonts="1">'
        '<p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId1"/></p:sldMasterIdLst>'
        f'<p:sldIdLst>{slide_ids}</p:sldIdLst>'
        f'<p:sldSz cx="{CX}" cy="{CY}"/><p:notesSz cx="6858000" cy="9144000"/></p:presentation>'
    )
    master = (
        f'{HDR}<p:sldMaster {NS}><p:cSld><p:bg><p:bgRef idx="1001"><a:schemeClr val="bg1"/></p:bgRef></p:bg>'
        f'<p:spTree>{EMPTY_TREE}</p:spTree></p:cSld>'
        '<p:clrMap bg1="lt1" tx1="dk1" bg2="lt2" tx2="dk2" accent1="accent1" accent2="accent2" '
        'accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" hlink="hlink" folHlink="folHlink"/>'
        '<p:sldLayoutIdLst><p:sldLayoutId id="2147483649" r:id="rId1"/></p:sldLayoutIdLst>'
        '<p:txStyles><p:titleStyle/><p:bodyStyle/><p:otherStyle/></p:txStyles></p:sldMaster>'
    )
    layout = (
        f'{HDR}<p:sldLayout {NS} type="blank" preserve="1"><p:cSld name="Blank">'
        f'<p:spTree>{EMPTY_TREE}</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sldLayout>'
    )
    core = (
        f'{HDR}<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
        'xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" '
        'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
        "<dc:title>Least-Squares Temporal Difference Learning</dc:title></cp:coreProperties>"
    )
    app = (
        f'{HDR}<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties">'
        f"<Application>Microsoft Office PowerPoint</Application><Slides>{n}</Slides></Properties>"
    )

    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types)
        z.writestr("_rels/.rels", rels(
            ("rId1", f"{REL}/officeDocument", "ppt/presentation.xml"),
            ("rId2", "http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties", "docProps/core.xml"),
            ("rId3", f"{REL}/extended-properties", "docProps/app.xml"),
        ))
        z.writestr("docProps/core.xml", core)
        z.writestr("docProps/app.xml", app)
        z.writestr("ppt/presentation.xml", presentation)
        z.writestr("ppt/_rels/presentation.xml.rels", rels(
            ("rId1", f"{REL}/slideMaster", "slideMasters/slideMaster1.xml"),
            ("rId2", f"{REL}/theme", "theme/theme1.xml"),
            ("rId3", f"{REL}/presProps", "presProps.xml"),
            ("rId4", f"{REL}/viewProps", "viewProps.xml"),
            *[(f"rId{i + 4}", f"{REL}/slide", f"slides/slide{i}.xml") for i in range(1, n + 1)],
            (f"rId{n + 5}", f"{REL}/tableStyles", "tableStyles.xml"),
        ))
        z.writestr("ppt/presProps.xml", f"{HDR}<p:presentationPr {NS}/>")
        z.writestr("ppt/viewProps.xml", f"{HDR}<p:viewPr {NS}/>")
        z.writestr("ppt/tableStyles.xml", f'{HDR}<a:tblStyleLst xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" def="{{5C22544A-7EE6-4342-B048-85BDC9FD1C3A}}"/>')
        z.writestr("ppt/theme/theme1.xml", theme())
        z.writestr("ppt/slideMasters/slideMaster1.xml", master)
        z.writestr("ppt/slideMasters/_rels/slideMaster1.xml.rels", rels(
            ("rId1", f"{REL}/slideLayout", "../slideLayouts/slideLayout1.xml"),
            ("rId2", f"{REL}/theme", "../theme/theme1.xml"),
        ))
        z.writestr("ppt/slideLayouts/slideLayout1.xml", layout)
        z.writestr("ppt/slideLayouts/_rels/slideLayout1.xml.rels", rels(
            ("rId1", f"{REL}/slideMaster", "../slideMasters/slideMaster1.xml"),
        ))
        for i, page in enumerate(pages, 1):
            z.writestr(f"ppt/slides/slide{i}.xml", slide_xml(i))
            z.writestr(f"ppt/slides/_rels/slide{i}.xml.rels", rels(
                ("rId1", f"{REL}/slideLayout", "../slideLayouts/slideLayout1.xml"),
                ("rId2", f"{REL}/image", f"../media/image{i}.png"),
            ))
            z.write(page, f"ppt/media/image{i}.png")


def main() -> None:
    if not PDF.exists():
        sys.exit(f"{PDF.name} not found; compile the .tex first")
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["pdftoppm", "-png", "-r", str(DPI), str(PDF), f"{tmp}/page"], check=True)
        pages = sorted(Path(tmp).glob("page-*.png"), key=lambda p: int(p.stem.split("-")[-1]))
        build(pages, PPTX)
    print(f"wrote {PPTX.name} ({len(pages)} slides)")


if __name__ == "__main__":
    main()
