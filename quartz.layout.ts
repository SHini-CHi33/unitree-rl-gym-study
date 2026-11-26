import { PageLayout, SharedLayout } from "./quartz/cfg"
import * as Component from "./quartz/components"

// 过滤 Explorer 中的 index、home 和 tags
const explorerFilter = (node: { slugSegment: string }) => {
  return (
    node.slugSegment !== "tags" &&
    node.slugSegment !== "index" &&
    node.slugSegment !== "home"
  )
}

// 判断当前页面是否不是首页
const notHome = ({ fileData }: { fileData: { slug?: string } }) =>
  fileData.slug !== "index"

// 全站共享组件
export const sharedPageComponents: SharedLayout = {
  head: Component.Head(),
  header: [],
  afterBody: [],
  footer: Component.Footer({
    links: {
      GitHub: "https://github.com/SHini-CHi33/unitree-rl-gym-study",
    },
  }),
}

// 普通笔记页面
export const defaultContentPageLayout: PageLayout = {
  beforeBody: [
    Component.ConditionalRender({
      component: Component.Breadcrumbs(),
      condition: notHome,
    }),
    Component.ConditionalRender({
      component: Component.ArticleTitle(),
      condition: notHome,
    }),
    Component.ConditionalRender({
      component: Component.ContentMeta(),
      condition: notHome,
    }),
    Component.ConditionalRender({
      component: Component.TagList(),
      condition: notHome,
    }),
  ],
  left: [
    Component.PageTitle(),
    Component.MobileOnly(Component.Spacer()),
    Component.Search(),
    Component.Darkmode(),
    Component.DesktopOnly(
      Component.Explorer({
        title: "学习目录",
        filterFn: (node) =>
          !["tags", "index", "home"].includes(node.slugSegment),
      }),
    ),
  ],
  right: [
    Component.ConditionalRender({
      component: Component.Graph(),
      condition: notHome,
    }),
    Component.ConditionalRender({
      component: Component.DesktopOnly(Component.TableOfContents()),
      condition: notHome,
    }),
    Component.ConditionalRender({
      component: Component.Backlinks(),
      condition: notHome,
    }),
  ],
}

// 文件夹和列表页面
export const defaultListPageLayout: PageLayout = {
  beforeBody: [
    Component.Breadcrumbs(),
    Component.ArticleTitle(),
    Component.ContentMeta(),
  ],
  left: [
    Component.PageTitle(),
    Component.MobileOnly(Component.Spacer()),
    Component.Search(),
    Component.Darkmode(),
    Component.DesktopOnly(
      Component.Explorer({
        filterFn: explorerFilter,
      }),
    ),
  ],
  right: [],
}