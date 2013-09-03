import os
import os.path as op

import numpy as np
import pandas as pd
import nibabel as nib


def write_workflow_report(workflow_name, report_template, report_dict):
    """Generic function to take write .rst files and convert to pdf/html.

    Accepts a report template and dictionary. Writes rst once with
    full paths for image files and generates a pdf, then strips
    leading path components and writes again, generating an html
    file that exepects to live in the same directory as report images.

    """
    from os.path import exists, basename
    from subprocess import check_output

    # Plug the values into the template for the pdf file
    report_rst_text = report_template % report_dict

    # Write the rst file and convert to pdf
    report_pdf_rst_file = "%s_pdf.rst" % workflow_name
    report_pdf_file = op.abspath("%s_report.pdf" % workflow_name)
    open(report_pdf_rst_file, "w").write(report_rst_text)
    check_output(["rst2pdf", report_pdf_rst_file, "-o", report_pdf_file])
    if not exists(report_pdf_file):
        raise RuntimeError

    # For images going into the html report, we want the path to be relative
    # (We expect to read the html page from within the datasink directory
    # containing the images.  So iteratate through and chop off leading path.
    for k, v in report_dict.items():
        if isinstance(v, str) and v.endswith(".png"):
            report_dict[k] = basename(v)

    # Write the another rst file and convert it to html
    report_html_rst_file = "%s_html.rst" % workflow_name
    report_html_file = op.abspath("%s_report.html" % workflow_name)
    report_rst_text = report_template % report_dict
    open(report_html_rst_file, "w").write(report_rst_text)
    check_output(["rst2html.py", report_html_rst_file, report_html_file])
    if not exists(report_html_file):
        raise RuntimeError

    # Return both report files as a list
    return [report_pdf_file, report_html_file]
