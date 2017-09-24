from textwrap import indent
import lyman

if __name__ == "__main__":

    project_info = lyman.frontend.ProjectInfo()
    experiment_info = lyman.frontend.ExperimentInfo()
    model_info = lyman.frontend.ModelInfo()

    project_traits = project_info.trait_get()
    model_traits = model_info.trait_get()
    experiment_traits = {k: v for
                         k, v in experiment_info.trait_get().items()
                         if k not in model_traits}

    groups = [("project", project_info, project_traits),
              ("experiment", experiment_info, experiment_traits),
              ("model", model_info, model_traits)]

    for group, group_info, group_traits in groups:
        with open("traits/{}.txt".format(group), "w") as fid:
            fid.write(".. glossary::\n\n")
            for trait, default in group_traits.items():
                fid.write(indent(trait, "   "))
                desc = group_info.trait(trait).desc
                fid.write(indent(desc, "    "))
                if "Automatically populated" not in desc:
                    default_text = "(Defaults to ``{}``)".format(default)
                    fid.write(indent(default_text, "    "))
                fid.write("\n\n")
